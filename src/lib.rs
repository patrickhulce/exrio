use smallvec::SmallVec;

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

mod attributes;
use attributes::{ImageAttributeHandler, IMAGE_HANDLERS};

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

    let pixels_f32 = match &layer.pixels_f32 {
        Some(pixels_f32) => pixels_f32.clone(),
        None => return None,
    };

    let mut channels_list = Vec::<AnyChannel<FlatSamples>>::new();

    for (index, channel) in pixels_f32.iter().enumerate() {
        let channel_name = match layer.channels.get(index) {
            Some(channel_name) => channel_name,
            None => return None,
        };

        channels_list.push(AnyChannel::new(
            channel_name.as_str(),
            FlatSamples::F32(channel.clone()),
        ));
    }

    let channels_builder = AnyChannels::sort(SmallVec::from_vec(channels_list));

    let image_with_channels = Image::from_channels(Vec2(*width, *height), channels_builder);

    let layer_out = Layer::new(
        Vec2(*width, *height),
        LayerAttributes::named(Text::from(name.as_str())),
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
    pixels_f32: Option<Vec<Vec<f32>>>,
}

fn layer_from_exr(exr_layer: Layer<AnyChannels<FlatSamples>>) -> ExrLayer {
    let name = exr_layer.attributes.layer_name.map(|name| name.to_string());
    let channels = exr_layer
        .channel_data
        .list
        .iter()
        .map(|channel| channel.name.to_string())
        .collect();
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
        width: Some(exr_layer.size.0),
        height: Some(exr_layer.size.1),
        pixels_f32,
    }
}

fn get_image_attributes_dict<'py>(
    py: Python<'py>,
    attributes: &ImageAttributes,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, handler) in attributes::IMAGE_HANDLERS {
        match (handler.getter)(attributes, py) {
            Some(value) => match value {
                Ok(value) => dict.set_item(key, value)?,
                Err(e) => return Err(e),
            },
            None => (),
        }
    }

    for (key, value) in attributes.other.iter() {
        match value {
            AttributeValue::Text(text) => dict.set_item(key.to_string(), text.to_string())?,
            AttributeValue::F64(f64) => dict.set_item(key.to_string(), f64)?,
            AttributeValue::F32(f32) => dict.set_item(key.to_string(), f32)?,
            AttributeValue::I32(i32) => dict.set_item(key.to_string(), i32)?,
            _ => (),
        }
    }

    Ok(dict)
}

fn set_image_attributes(attributes: &mut ImageAttributes, dict: &Bound<PyDict>) -> PyResult<()> {
    for (key, handler) in attributes::IMAGE_HANDLERS {
        match dict.contains(key) {
            Ok(true) => (handler.setter)(attributes, dict)?,
            _ => (),
        }
    }

    for (key, value) in dict.iter() {
        let key_str = key.to_string();
        if attributes::IMAGE_HANDLERS
            .iter()
            .all(|(k, _)| k != &key_str)
        {
            let key_text = Text::from(key_str.as_str());
            let value_text = Text::from(value.to_string().as_str());
            let attribute_value = AttributeValue::Text(value_text);
            attributes.other.insert(key_text, attribute_value);
        }
    }

    Ok(())
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
            pixels_f32: None,
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

    fn pixels_f32<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyArray1<f32>>>>> {
        let pixels_32 = self.pixels_f32.clone().map(|channels| {
            channels
                .iter()
                .map(|channel| vec_to_numpy_array(py, channel))
                .collect()
        });

        Ok(pixels_32)
    }

    fn with_channel_f32<'py>(
        &mut self,
        py: Python<'py>,
        channel: String,
        pixels: Bound<'py, PyArray1<f32>>,
    ) -> PyResult<()> {
        if self.width.is_none() || self.height.is_none() {
            return Err(PyIOError::new_err(
                "Layer width and height must be set before adding a channel",
            ));
        }

        let width = self.width.unwrap();
        let height = self.height.unwrap();

        let expected_pixels = width * height;
        let actual_pixels = match pixels.len() {
            Ok(len) => len,
            Err(e) => return Err(e),
        };

        if expected_pixels != actual_pixels {
            return Err(PyIOError::new_err(
                "Width * height must match the number of pixels",
            ));
        }

        self.channels.push(channel);
        self.width = Some(width);
        self.height = Some(height);

        let pixels_to_add = match pixels.to_vec() {
            Ok(vec) => vec,
            Err(e) => return Err(PyIOError::new_err(e.to_string())),
        };

        if self.pixels_f32.is_none() {
            self.pixels_f32 = Some(vec![pixels_to_add]);
        } else {
            self.pixels_f32.as_mut().unwrap().push(pixels_to_add);
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
        get_image_attributes_dict(py, &self.attributes)
    }

    fn with_attributes(&mut self, dict: &Bound<PyDict>) -> PyResult<()> {
        set_image_attributes(&mut self.attributes, dict)
    }

    fn layers(&self) -> Vec<ExrLayer> {
        self.layers.clone()
    }

    fn with_layer(&mut self, layer: ExrLayer) {
        self.layers.push(layer);
    }

    fn save_to_path<'py>(&self, py: Python<'py>, file_path: &str) -> PyResult<()> {
        let first_layer = self.layers.first().unwrap();
        let rust_layers: Vec<Layer<AnyChannels<FlatSamples>>> = self
            .layers
            .iter()
            .flat_map(|layer| to_rust_layer(layer))
            .collect();

        let mut attributes = self.attributes.clone();
        attributes.display_window.size.0 = first_layer.width.unwrap();
        attributes.display_window.size.1 = first_layer.height.unwrap();

        Image::from_layers(attributes, rust_layers)
            .write()
            .to_file(file_path)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(())
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

        Ok(ExrImage {
            layers,
            attributes: image.attributes,
        })
    }
}

#[pymodule]
fn exrio<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<ExrImage>()?;
    m.add_class::<ExrLayer>()?;
    Ok(())
}
