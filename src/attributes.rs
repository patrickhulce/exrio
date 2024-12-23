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

pub type ImageAttributeGetterFn =
    for<'py> fn(&ImageAttributes, Python<'py>) -> Option<PyResult<Py<PyAny>>>;
pub type ImageAttributeSetterFn =
    for<'py> fn(&mut ImageAttributes, &'py Bound<'py, PyDict>) -> PyResult<()>;

pub struct ImageAttributeHandler {
    pub getter: ImageAttributeGetterFn,
    pub setter: ImageAttributeSetterFn,
}

fn extract_int(dict: &Bound<PyDict>, key: &str) -> PyResult<i32> {
    match dict.get_item(key)? {
        Some(value) => value
            .extract::<i32>()
            .map_err(|_| PyIOError::new_err(format!("{} invalid", key))),
        None => Err(PyIOError::new_err(format!("{} not found", key))),
    }
}

fn extract_float(dict: &Bound<PyDict>, key: &str) -> PyResult<f32> {
    match dict.get_item(key)? {
        Some(value) => value
            .extract::<f32>()
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

pub const IMAGE_HANDLERS: &[(&str, ImageAttributeHandler)] = &[
    (
        "display_window.position.0",
        ImageAttributeHandler {
            getter: |attrs, py| Some(attrs.display_window.position.0.into_py_any(py)),
            setter: |attrs, dict| {
                match extract_int(dict, "display_window.position.0") {
                    Ok(value) => attrs.display_window.position.0 = value,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "display_window.position.1",
        ImageAttributeHandler {
            getter: |attrs, py| Some(attrs.display_window.position.1.into_py_any(py)),
            setter: |attrs, dict| {
                match extract_int(dict, "display_window.position.1") {
                    Ok(value) => attrs.display_window.position.1 = value,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "display_window.size.0",
        ImageAttributeHandler {
            getter: |attrs, py| Some(attrs.display_window.size.0.into_py_any(py)),
            setter: |attrs, dict| {
                match extract_int(dict, "display_window.size.0") {
                    Ok(value) => attrs.display_window.size.0 = value as usize,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "display_window.size.1",
        ImageAttributeHandler {
            getter: |attrs, py| Some(attrs.display_window.size.1.into_py_any(py)),
            setter: |attrs, dict| {
                match extract_int(dict, "display_window.size.1") {
                    Ok(value) => attrs.display_window.size.1 = value as usize,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "pixel_aspect",
        ImageAttributeHandler {
            getter: |attrs, py| Some(attrs.pixel_aspect.into_py_any(py)),
            setter: |attrs, dict| {
                match extract_float(dict, "pixel_aspect") {
                    Ok(value) => attrs.pixel_aspect = value,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "chromaticities.red.0",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.chromaticities.map(|c| c.red.0.into_py_any(py)),
            setter: |attrs, dict| {
                let mut chromaticities = get_chromaticities_or_default(attrs);
                match extract_float(dict, "chromaticities.red.0") {
                    Ok(value) => chromaticities.red.0 = value,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "chromaticities.red.1",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.chromaticities.map(|c| c.red.1.into_py_any(py)),
            setter: |attrs, dict| {
                let mut chromaticities = get_chromaticities_or_default(attrs);
                match extract_float(dict, "chromaticities.red.1") {
                    Ok(value) => chromaticities.red.1 = value,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "chromaticities.green.0",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.chromaticities.map(|c| c.green.0.into_py_any(py)),
            setter: |attrs, dict| {
                let mut chromaticities = get_chromaticities_or_default(attrs);
                match extract_float(dict, "chromaticities.green.0") {
                    Ok(value) => chromaticities.green.0 = value,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "chromaticities.green.1",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.chromaticities.map(|c| c.green.1.into_py_any(py)),
            setter: |attrs, dict| {
                let mut chromaticities = get_chromaticities_or_default(attrs);
                match extract_float(dict, "chromaticities.green.1") {
                    Ok(value) => chromaticities.green.1 = value,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "chromaticities.blue.0",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.chromaticities.map(|c| c.blue.0.into_py_any(py)),
            setter: |attrs, dict| {
                let mut chromaticities = get_chromaticities_or_default(attrs);
                match extract_float(dict, "chromaticities.blue.0") {
                    Ok(value) => chromaticities.blue.0 = value,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "chromaticities.blue.1",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.chromaticities.map(|c| c.blue.1.into_py_any(py)),
            setter: |attrs, dict| {
                let mut chromaticities = get_chromaticities_or_default(attrs);
                match extract_float(dict, "chromaticities.blue.1") {
                    Ok(value) => chromaticities.blue.1 = value,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "chromaticities.white.0",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.chromaticities.map(|c| c.white.0.into_py_any(py)),
            setter: |attrs, dict| {
                let mut chromaticities = get_chromaticities_or_default(attrs);
                match extract_float(dict, "chromaticities.white.0") {
                    Ok(value) => chromaticities.white.0 = value,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "chromaticities.white.1",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.chromaticities.map(|c| c.white.1.into_py_any(py)),
            setter: |attrs, dict| {
                let mut chromaticities = get_chromaticities_or_default(attrs);
                match extract_float(dict, "chromaticities.white.1") {
                    Ok(value) => chromaticities.white.1 = value,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "timecode.hours",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.time_code.map(|t| t.hours.into_py_any(py)),
            setter: |attrs, dict| {
                let mut timecode = get_timecode_or_default(attrs);
                match extract_int(dict, "timecode.hours") {
                    Ok(value) => timecode.hours = value as u8,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "timecode.minutes",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.time_code.map(|t| t.minutes.into_py_any(py)),
            setter: |attrs, dict| {
                let mut timecode = get_timecode_or_default(attrs);
                match extract_int(dict, "timecode.minutes") {
                    Ok(value) => timecode.minutes = value as u8,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "timecode.seconds",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.time_code.map(|t| t.seconds.into_py_any(py)),
            setter: |attrs, dict| {
                let mut timecode = get_timecode_or_default(attrs);
                match extract_int(dict, "timecode.seconds") {
                    Ok(value) => timecode.seconds = value as u8,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
    (
        "timecode.frame",
        ImageAttributeHandler {
            getter: |attrs, py| attrs.time_code.map(|t| t.frame.into_py_any(py)),
            setter: |attrs, dict| {
                let mut timecode = get_timecode_or_default(attrs);
                match extract_int(dict, "timecode.frame") {
                    Ok(value) => timecode.frame = value as u8,
                    Err(e) => return Err(e),
                }
                Ok(())
            },
        },
    ),
];
