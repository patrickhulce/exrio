use exr::prelude::read::any_channels::ReadAnyChannels;
use exr::prelude::read::layers::ReadAllLayers;
use exr::prelude::read::samples::ReadFlatSamples;
use exr::prelude::*;
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArrayDyn,
};
use pyo3::{
    exceptions::PyIOError,
    pyclass, pymethods, pymodule,
    types::{PyAnyMethods, PyBytes, PyDict, PyDictMethods, PyModule, PyModuleMethods},
    Bound, FromPyObject, PyAny, PyObject, PyResult, Python,
};

fn get_image_reader() -> ReadImage<fn(f64), ReadAllLayers<ReadAnyChannels<ReadFlatSamples>>> {
    let image = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .all_layers()
        .all_attributes();

    image
}

fn vec_to_numpy_array<'py>(py: Python<'py>, vec: &Vec<f32>) -> Bound<'py, PyArray1<f32>> {
    PyArray1::from_iter(py, vec.iter().map(|value| *value as f32))
}

#[pyclass]
#[derive(Clone)]
struct ExrLayer {
    name: Option<String>,
    channels: Option<Vec<String>>,
    width: Option<u32>,
    height: Option<u32>,
    pixels_f32: Option<Vec<Vec<f32>>>,
}

fn layer_from_exr(exr_layer: Layer<AnyChannels<FlatSamples>>) -> ExrLayer {
    let name = exr_layer.attributes.layer_name.map(|name| name.to_string());
    let channels = Some(
        exr_layer
            .channel_data
            .list
            .iter()
            .map(|channel| channel.name.to_string())
            .collect(),
    );
    let pixels_f32 = Some(
        exr_layer
            .channel_data
            .list
            .iter()
            .map(|channel| channel.sample_data.values_as_f32().collect())
            .collect(),
    );

    ExrLayer {
        name,
        channels,
        width: Some(exr_layer.size.0 as u32),
        height: Some(exr_layer.size.1 as u32),
        pixels_f32,
    }
}

#[pymethods]
impl ExrLayer {
    #[new]
    #[pyo3(signature = (name = None, channels = None))]
    fn new(name: Option<String>, channels: Option<Vec<String>>) -> Self {
        Self {
            name,
            channels,
            width: None,
            height: None,
            pixels_f32: None,
        }
    }

    fn name(&self) -> Option<String> {
        self.name.clone()
    }

    fn channels(&self) -> Option<Vec<String>> {
        self.channels.clone()
    }

    fn width(&self) -> Option<u32> {
        self.width
    }

    fn height(&self) -> Option<u32> {
        self.height
    }

    fn pixels_f32<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyArray1<f32>>>>> {
        let pixels_32 = self.pixels_f32.clone().map(|channels| {
            channels
                .iter()
                .map(|channel| vec_to_numpy_array(py, channel))
                .collect()
        });

        Ok(pixels_32)
    }
}

#[pyclass]
struct ExrImage {
    layers: Vec<ExrLayer>,
}

#[pymethods]
impl ExrImage {
    fn layers(&self) -> Vec<ExrLayer> {
        self.layers.clone()
    }

    #[staticmethod]
    fn load_from_path(file_path: &str) -> PyResult<ExrImage> {
        let image = match get_image_reader().from_file(file_path) {
            Ok(image) => image,
            Err(e) => return Err(PyIOError::new_err(e.to_string())),
        };

        let mut layers: Vec<ExrLayer> = Vec::new();
        for layer in image.layer_data {
            layers.push(layer_from_exr(layer));
        }

        Ok(ExrImage { layers })
    }
}

#[pymodule]
fn exrio<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<ExrImage>()?;

    Ok(())
}
