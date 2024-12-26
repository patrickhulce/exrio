use std::{collections::HashMap, fmt::format};

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
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, Default)]
struct SerializableAttrValue {
    pub values_i32: Option<Vec<i32>>,
    pub values_f32: Option<Vec<f32>>,
    pub values_text: Option<Vec<String>>,
    pub values_u8: Option<Vec<u8>>,
    pub values_bool: Option<Vec<bool>>,
}

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
        name: "timecode",
        to_python: |value, py| match value {
            AttributeValue::TimeCode(timecode) => {
                let serializable_value = SerializableAttrValue {
                    values_i32: Some(vec![
                        timecode.hours as i32,
                        timecode.minutes as i32,
                        timecode.seconds as i32,
                        timecode.frame as i32,
                    ]),
                    values_f32: None,
                    values_text: None,
                    values_u8: Some(timecode.binary_groups.to_vec()),
                    values_bool: Some(vec![
                        timecode.drop_frame,
                        timecode.color_frame,
                        timecode.field_phase,
                        timecode.binary_group_flags[0],
                        timecode.binary_group_flags[1],
                        timecode.binary_group_flags[2],
                    ]),
                };

                match serde_json::to_string(&serializable_value) {
                    Ok(value) => Some(format!("timecode:{}", value).into_py_any(py)),
                    Err(e) => return Some(Err(PyIOError::new_err(format!("{} invalid", e)))),
                }
            }
            _ => None,
        },
        from_python: |value| match value.extract::<String>() {
            Ok(value) => {
                if !value.starts_with("timecode:") {
                    return Err(PyIOError::new_err("Invalid timecode"));
                }

                let value = value[8..].to_string();
                match serde_json::from_str::<SerializableAttrValue>(&value) {
                    Ok(value) => {
                        let values_i32 = value.values_i32.unwrap();
                        let values_bool = value.values_bool.unwrap();
                        let values_u8 = value.values_u8.unwrap();

                        Ok(AttributeValue::TimeCode(TimeCode {
                            hours: values_i32[0] as u8,
                            minutes: values_i32[1] as u8,
                            seconds: values_i32[2] as u8,
                            frame: values_i32[3] as u8,
                            drop_frame: values_bool[0],
                            color_frame: values_bool[1],
                            field_phase: values_bool[2],
                            binary_group_flags: [values_bool[3], values_bool[4], values_bool[5]],
                            binary_groups: values_u8.try_into().unwrap(),
                        }))
                    }
                    Err(e) => Err(PyIOError::new_err(format!("{} invalid", e))),
                }
            }
            Err(e) => Err(PyIOError::new_err(format!("{} invalid", e))),
        },
    },
    AttributeValueHandler {
        name: "integer",
        to_python: |value, py| match value {
            AttributeValue::I32(integer) => Some(integer.into_py_any(py)),
            _ => None,
        },
        from_python: |value| match value.extract::<i32>() {
            Ok(value) => Ok(AttributeValue::I32(value)),
            Err(e) => Err(PyIOError::new_err(format!("{} invalid", e))),
        },
    },
    AttributeValueHandler {
        name: "intvec2",
        to_python: |value, py| match value {
            AttributeValue::IntVec2(vec) => Some([vec.0, vec.1].into_py_any(py)),
            _ => None,
        },
        from_python: |value| match value.extract::<Vec<i32>>() {
            Ok(value) => Ok(AttributeValue::IntVec2(Vec2(value[0], value[1]))),
            Err(e) => Err(PyIOError::new_err(format!("{} invalid", e))),
        },
    },
    AttributeValueHandler {
        name: "floatvec2",
        to_python: |value, py| match value {
            AttributeValue::FloatVec2(vec) => Some([vec.0, vec.1].into_py_any(py)),
            _ => None,
        },
        from_python: |value| match value.extract::<Vec<f32>>() {
            Ok(value) => Ok(AttributeValue::FloatVec2(Vec2(value[0], value[1]))),
            Err(e) => Err(PyIOError::new_err(format!("{} invalid", e))),
        },
    },
    AttributeValueHandler {
        name: "chromaticities",
        to_python: |value, py| match value {
            AttributeValue::Chromaticities(chromaticities) => Some(
                format!(
                    "chroma:{:?}-{:?}-{:?}-{:?}-{:?}-{:?}-{:?}-{:?}",
                    chromaticities.red.0,
                    chromaticities.red.1,
                    chromaticities.green.0,
                    chromaticities.green.1,
                    chromaticities.blue.0,
                    chromaticities.blue.1,
                    chromaticities.white.0,
                    chromaticities.white.1
                )
                .into_py_any(py),
            ),
            _ => None,
        },
        from_python: |value| match value.extract::<String>() {
            Ok(value) => {
                if !value.starts_with("chroma:") {
                    return Err(PyIOError::new_err("Invalid chromaticities"));
                }

                let values = value
                    .replace("chroma:", "")
                    .split('-')
                    .flat_map(|s| s.parse::<f32>())
                    .collect::<Vec<f32>>();

                if values.len() != 8 {
                    return Err(PyIOError::new_err("Invalid chromaticities"));
                }

                Ok(AttributeValue::Chromaticities(Chromaticities {
                    red: Vec2(values[0], values[1]),
                    green: Vec2(values[2], values[3]),
                    blue: Vec2(values[4], values[5]),
                    white: Vec2(values[6], values[7]),
                }))
            }
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
