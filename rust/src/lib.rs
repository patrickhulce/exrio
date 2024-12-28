use std::collections::HashMap;

use smallvec::{Array, SmallVec};

use exr::prelude::read::any_channels::ReadAnyChannels;
use exr::prelude::read::layers::ReadAllLayers;
use exr::prelude::read::samples::ReadFlatSamples;
use exr::prelude::*;
use half::f16;
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArrayDyn, PyReadwriteArray1, PyReadwriteArrayDyn,
};
use pyo3::{
    exceptions::PyIOError,
    pyclass, pymethods, pymodule,
    types::{PyAnyMethods, PyBytes, PyDict, PyDictMethods, PyModule, PyModuleMethods},
    Bound, FromPyObject, Py, PyAny, PyErr, PyObject, PyResult, Python,
};
use std::io::{self, BufWriter, Cursor, Write};
use std::vec::Vec;

fn get_inmemory_writer() -> BufWriter<Cursor<Vec<u8>>> {
    let buffer = Vec::new();
    let cursor = Cursor::new(buffer);

    BufWriter::new(cursor)
}

mod pyattributes;
use pyattributes::{from_python, to_python, AttributeValueHandler, IMAGE_HANDLERS};

mod attributes;
use attributes::{attributes_from_image, attributes_from_layer, image_attributes_from_attributes};

fn get_image_reader() -> ReadImage<fn(f64), ReadAllLayers<ReadAnyChannels<ReadFlatSamples>>> {
    let image = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .all_layers()
        .all_attributes();

    image
}

fn vec_to_numpy_array<'py>(py: Python<'py>, array_data: &PixelData) -> Bound<'py, PyAny> {
    match array_data {
        PixelData::F32(vec) => {
            PyArray1::from_iter(py, vec.iter().map(|value| *value as f32)).into_any()
        }
        PixelData::F16(vec) => {
            PyArray1::from_iter(py, vec.iter().map(|value| *value as f16)).into_any()
        }
        PixelData::U32(vec) => {
            PyArray1::from_iter(py, vec.iter().map(|value| *value as u32)).into_any()
        }
    }
}

fn to_rust_layer(layer: &ExrLayer) -> Option<Layer<AnyChannels<FlatSamples>>> {
    let name = match &layer.name {
        Some(name) => name,
        None => return None,
    };

    let width = match &layer.width {
        Some(width) => width,
        None => return None,
    };

    let height = match &layer.height {
        Some(height) => height,
        None => return None,
    };

    let pixels = match &layer.pixels {
        Some(pixels) => pixels.clone(),
        None => return None,
    };

    let mut channels_list = Vec::<AnyChannel<FlatSamples>>::new();

    for (index, channel) in pixels.iter().enumerate() {
        let channel_name = match layer.channels.get(index) {
            Some(channel_name) => channel_name,
            None => return None,
        };

        let samples = match channel {
            PixelData::F32(vec) => FlatSamples::F32(vec.clone()),
            PixelData::F16(vec) => FlatSamples::F16(vec.clone()),
            PixelData::U32(vec) => FlatSamples::U32(vec.clone()),
        };

        channels_list.push(AnyChannel::new(channel_name.as_str(), samples));
    }

    let channels_builder = AnyChannels::sort(SmallVec::from_vec(channels_list));

    let image_with_channels = Image::from_channels(Vec2(*width, *height), channels_builder);

    let mut attributes = LayerAttributes::named(Text::from(name.as_str()));
    let _ = attributes::layer_attributes_from_attributes(&mut attributes, &layer.attributes);

    let layer_out = Layer::new(
        Vec2(*width, *height),
        attributes,
        Encoding::FAST_LOSSLESS,
        image_with_channels.layer_data.channel_data,
    );

    Some(layer_out)
}

#[pyclass]
#[derive(Clone)]
struct ExrLayer {
    name: Option<String>,
    channels: Vec<String>,
    width: Option<usize>,
    height: Option<usize>,
    pixels: Option<Vec<PixelData>>,
    attributes: HashMap<Text, AttributeValue>,
}

fn layer_from_exr(exr_layer: Layer<AnyChannels<FlatSamples>>) -> ExrLayer {
    let attributes = attributes_from_layer(&exr_layer.attributes);
    let name = exr_layer.attributes.layer_name.map(|name| name.to_string());
    let channels = exr_layer
        .channel_data
        .list
        .iter()
        .map(|channel| channel.name.to_string())
        .collect();
    let pixels = Some(
        exr_layer
            .channel_data
            .list
            .iter()
            .map(|channel| match &channel.sample_data {
                FlatSamples::F32(vec) => PixelData::F32(vec.clone()),
                FlatSamples::F16(vec) => PixelData::F16(vec.clone()),
                FlatSamples::U32(vec) => PixelData::U32(vec.clone()),
            })
            .collect(),
    );

    ExrLayer {
        name,
        channels,
        width: Some(exr_layer.size.0),
        height: Some(exr_layer.size.1),
        pixels,
        attributes,
    }
}

#[derive(Clone)]
enum PixelData {
    F16(Vec<f16>),
    F32(Vec<f32>),
    U32(Vec<u32>),
}

fn _validate_width_height_pixels(
    width_option: Option<usize>,
    height_option: Option<usize>,
    pixels: &PixelData,
) -> PyResult<()> {
    if width_option.is_none() || height_option.is_none() {
        return Err(PyIOError::new_err(
            "Layer width and height must be set before adding a channel",
        ));
    }

    let width = width_option.unwrap();
    let height = height_option.unwrap();
    let expected_pixels = width * height;
    let actual_pixels = match pixels {
        PixelData::F32(vec) => vec.len(),
        PixelData::F16(vec) => vec.len(),
        PixelData::U32(vec) => vec.len(),
    };

    if expected_pixels != actual_pixels {
        return Err(PyIOError::new_err(
            "Width * height must match the number of pixels",
        ));
    }

    Ok(())
}

fn convert_numpy_array<'py>(py: Python<'py>, array: &Bound<'py, PyAny>) -> PyResult<PixelData> {
    if let Ok(array) = array.extract::<PyReadonlyArray1<f32>>() {
        return Ok(PixelData::F32(array.to_vec()?));
    }
    if let Ok(array) = array.extract::<PyReadonlyArray1<f16>>() {
        return Ok(PixelData::F16(array.to_vec()?));
    }
    if let Ok(array) = array.extract::<PyReadonlyArray1<u32>>() {
        return Ok(PixelData::U32(array.to_vec()?));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Unsupported array type",
    ))
}

#[pymethods]
impl ExrLayer {
    #[new]
    #[pyo3(signature = (name = None))]
    fn new(name: Option<String>) -> Self {
        Self {
            name,
            channels: Vec::new(),
            width: None,
            height: None,
            pixels: None,
            attributes: HashMap::new(),
        }
    }

    fn name(&self) -> Option<String> {
        self.name.clone()
    }

    fn channels(&self) -> Vec<String> {
        self.channels.clone()
    }

    fn width(&self) -> Option<usize> {
        self.width
    }

    fn with_width(&mut self, width: usize) {
        self.width = Some(width);
    }

    fn height(&self) -> Option<usize> {
        self.height
    }

    fn with_height(&mut self, height: usize) {
        self.height = Some(height);
    }

    fn pixels<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        let pixels = self.pixels.clone().map(|channels| {
            channels
                .iter()
                .map(|channel| vec_to_numpy_array(py, channel))
                .collect()
        });

        Ok(pixels)
    }

    fn with_channel<'py>(
        &mut self,
        py: Python<'py>,
        channel: String,
        pixels: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let array_data = convert_numpy_array(py, pixels)?;

        if let Err(e) = _validate_width_height_pixels(self.width, self.height, &array_data) {
            return Err(e);
        }

        if self.pixels.is_none() {
            self.pixels = Some(vec![array_data]);
        } else {
            self.pixels.as_mut().unwrap().push(array_data);
        }
        self.channels.push(channel);

        Ok(())
    }

    fn attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pyattributes::pydict_from_attributes(py, &self.attributes)
    }

    fn with_attributes<'py>(&mut self, py: Python<'py>, dict: &Bound<PyDict>) -> PyResult<()> {
        match pyattributes::attributes_from_pydict(py, dict) {
            Ok(attributes) => {
                for (key, value) in attributes.iter() {
                    self.attributes.insert(key.clone(), value.clone());
                }
            }
            Err(e) => return Err(e),
        }

        Ok(())
    }
}

#[pyclass]
struct ExrImage {
    layers: Vec<ExrLayer>,
    attributes: ImageAttributes,
}

#[pymethods]
impl ExrImage {
    #[new]
    fn new() -> Self {
        Self {
            layers: Vec::new(),
            attributes: ImageAttributes::new(IntegerBounds::from_dimensions((0, 0))),
        }
    }

    fn attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pyattributes::pydict_from_attributes(py, &attributes_from_image(&self.attributes))
    }

    fn with_attributes<'py>(&mut self, py: Python<'py>, dict: &Bound<PyDict>) -> PyResult<()> {
        match pyattributes::attributes_from_pydict(py, dict) {
            Ok(attributes) => image_attributes_from_attributes(&mut self.attributes, &attributes)
                .map_err(|e| PyIOError::new_err(e.to_string())),
            Err(e) => return Err(e),
        }
    }

    fn layers(&self) -> Vec<ExrLayer> {
        self.layers.clone()
    }

    fn with_layer(&mut self, layer: ExrLayer) {
        self.layers.push(layer);
    }

    fn save_to_buffer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let first_layer = self.layers.first().unwrap();
        let rust_layers: Vec<Layer<AnyChannels<FlatSamples>>> = self
            .layers
            .iter()
            .flat_map(|layer| to_rust_layer(layer))
            .collect();

        let mut attributes = self.attributes.clone();
        attributes.display_window.size.0 = first_layer.width.unwrap();
        attributes.display_window.size.1 = first_layer.height.unwrap();

        let image = Image::from_layers(attributes, rust_layers);
        let mut writer = get_inmemory_writer();
        match image.write().to_buffered(&mut writer) {
            Ok(_) => (),
            Err(e) => return Err(PyIOError::new_err(e.to_string())),
        }

        match writer.into_inner() {
            Ok(buffer) => Ok(PyBytes::new(py, buffer.into_inner().as_slice())),
            Err(e) => Err(PyIOError::new_err(e.to_string())),
        }
    }

    #[staticmethod]
    fn load_from_buffer<'py>(py: Python<'py>, buffer: &Bound<'py, PyBytes>) -> PyResult<ExrImage> {
        let bytes: &[u8] = buffer.extract::<&[u8]>()?;
        let cursor = Cursor::new(bytes);
        let image = match get_image_reader().from_buffered(cursor) {
            Ok(image) => image,
            Err(e) => return Err(PyIOError::new_err(e.to_string())),
        };

        let mut layers: Vec<ExrLayer> = Vec::new();
        for layer in image.layer_data {
            layers.push(layer_from_exr(layer));
        }

        Ok(ExrImage {
            layers,
            attributes: image.attributes,
        })
    }
}

#[pymodule]
#[pyo3(name = "_rust")]
fn exrio<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<ExrImage>()?;
    m.add_class::<ExrLayer>()?;
    Ok(())
}
