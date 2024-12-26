use std::collections::HashMap;

use attribute::Chromaticities;
use exr::meta::attribute::TimeCode;
use exr::prelude::*;
use pyo3::{
    conversion::{IntoPyObject, IntoPyObjectExt},
    exceptions::PyIOError,
    pyclass, pymethods, pymodule,
    types::{PyAnyMethods, PyBytes, PyDict, PyDictMethods, PyModule, PyModuleMethods},
    Bound, FromPyObject, Py, PyAny, PyErr, PyObject, PyResult, Python,
};

pub type AttributeValueSerializeFn =
    for<'py> fn(&AttributeValue, Python<'py>) -> Option<PyResult<Py<PyAny>>>;
pub type AttributeValueDeserializeFn =
    for<'py> fn(&'py Bound<'py, PyAny>) -> PyResult<AttributeValue>;

pub struct AttributeValueHandler {
    pub name: &'static str,
    pub to_python: AttributeValueSerializeFn,
    pub from_python: AttributeValueDeserializeFn,
}

fn extract_int(dict: &Bound<PyDict>, key: &str) -> PyResult<i32> {
    match dict.get_item(key)? {
        Some(value) => value
            .extract::<i32>()
            .map_err(|_| PyIOError::new_err(format!("{} invalid", key))),
        None => Err(PyIOError::new_err(format!("{} not found", key))),
    }
}

fn get_chromaticities_or_default(attrs: &mut ImageAttributes) -> Chromaticities {
    let chromaticities = match attrs.chromaticities {
        Some(chromaticities) => chromaticities,
        None => {
            attrs.chromaticities = Some(Chromaticities {
                red: Vec2(0.64, 0.33),
                green: Vec2(0.3, 0.6),
                blue: Vec2(0.15, 0.06),
                white: Vec2(0.3127, 0.329),
            });
            return attrs.chromaticities.unwrap();
        }
    };
    attrs.chromaticities = Some(chromaticities);
    chromaticities
}

fn get_timecode_or_default(attrs: &mut ImageAttributes) -> TimeCode {
    let timecode = match attrs.time_code {
        Some(timecode) => timecode,
        None => TimeCode {
            hours: 0,
            minutes: 0,
            seconds: 0,
            frame: 0,
            drop_frame: false,
            color_frame: false,
            field_phase: false,
            binary_group_flags: [false, false, false],
            binary_groups: [0, 0, 0, 0, 0, 0, 0, 0],
        },
    };

    attrs.time_code = Some(timecode);
    timecode
}

pub const IMAGE_HANDLERS: &[AttributeValueHandler] = &[
    AttributeValueHandler {
        name: "f32",
        to_python: |value, py| match value {
            AttributeValue::F32(f32) => Some(f32.into_py_any(py)),
            _ => None,
        },
        from_python: |value| match value.extract::<f32>() {
            Ok(value) => Ok(AttributeValue::F32(value)),
            Err(e) => Err(PyIOError::new_err(format!("{} invalid", e))),
        },
    },
    AttributeValueHandler {
        name: "text",
        to_python: |value, py| match value {
            AttributeValue::Text(text) => Some(text.to_string().into_py_any(py)),
            _ => None,
        },
        from_python: |value| match value.extract::<String>() {
            Ok(value) => Ok(AttributeValue::Text(Text::from(value.as_str()))),
            Err(e) => Err(PyIOError::new_err(format!("{} invalid", e))),
        },
    },
    AttributeValueHandler {
        name: "integer_bounds",
        to_python: |value, py| match value {
            AttributeValue::IntegerBounds(bounds) => Some(
                format!(
                    "{:?}-{:?}-{:?}-{:?}",
                    bounds.position.0, bounds.position.1, bounds.size.0, bounds.size.1
                )
                .into_py_any(py),
            ),
            _ => None,
        },
        from_python: |value| match value.extract::<String>() {
            Ok(value) => {
                let values = value
                    .split('-')
                    .flat_map(|s| s.parse::<i32>())
                    .collect::<Vec<i32>>();

                if values.len() != 4 {
                    return Err(PyIOError::new_err("Invalid integer bounds"));
                }

                Ok(AttributeValue::IntegerBounds(IntegerBounds {
                    position: Vec2(values[0], values[1]),
                    size: Vec2(values[2] as usize, values[3] as usize),
                }))
            }
            Err(e) => Err(PyIOError::new_err(format!("{} invalid", e))),
        },
    },
];

pub fn to_python(key: &str, value: &AttributeValue, py: Python) -> PyResult<Py<PyAny>> {
    let mut last_error: Option<PyErr> = None;
    for handler in IMAGE_HANDLERS {
        match (handler.to_python)(value, py) {
            Some(value) => match value {
                Ok(value) => return Ok(value),
                Err(e) => last_error = Some(e),
            },
            None => (),
        }
    }

    let mut debug_string = String::new();
    for handler in IMAGE_HANDLERS {
        debug_string.push_str(handler.name);
        debug_string.push_str(", ");
    }

    if last_error.is_some() {
        debug_string.push_str(&last_error.unwrap().value(py).to_string());
    }

    Err(PyIOError::new_err(format!(
        "No matching attribute value serializer for {}. Last error: {}",
        key, debug_string
    )))
}

pub fn from_python<'py>(
    key: &str,
    value: &Bound<'py, PyAny>,
    py: Python<'py>,
) -> PyResult<AttributeValue> {
    let mut last_error: Option<PyErr> = None;
    for handler in IMAGE_HANDLERS {
        match (handler.from_python)(value) {
            Ok(value) => return Ok(value),
            Err(e) => last_error = Some(e),
        }
    }

    let mut debug_string = String::new();
    for handler in IMAGE_HANDLERS {
        debug_string.push_str(handler.name);
        debug_string.push_str(", ");
    }

    if last_error.is_some() {
        debug_string.push_str(&last_error.unwrap().value(py).to_string());
    }

    Err(PyIOError::new_err(format!(
        "No matching attribute value deserializer for {}. Last error: {}",
        key, debug_string
    )))
}

pub fn pydict_from_attributes<'py>(
    py: Python<'py>,
    attributes: &HashMap<Text, AttributeValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in attributes.iter() {
        let py_value = to_python(key.to_string().as_str(), value, py)?;
        dict.set_item(key.to_string(), py_value)?;
    }
    Ok(dict)
}

pub fn attributes_from_pydict<'py>(
    py: Python<'py>,
    pydict: &Bound<'py, PyDict>,
) -> PyResult<HashMap<Text, AttributeValue>> {
    let mut attributes = HashMap::new();

    for (key, value) in pydict.iter() {
        let key_str = key.to_string();
        match from_python(key_str.as_str(), &value, py) {
            Ok(attribute_value) => {
                attributes.insert(Text::from(key_str.as_str()), attribute_value);
            }
            Err(e) => return Err(e),
        };
    }

    Ok(attributes)
}
