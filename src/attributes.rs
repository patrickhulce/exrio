use std::collections::HashMap;

use exr::{
    meta::attribute::{
        Chromaticities, EnvironmentMap, KeyCode, Matrix4x4, Preview, Rational, TimeCode,
    },
    prelude::{
        AttributeValue, ImageAttributes, IntegerBounds, LayerAttributes, Result, Text, Vec2,
    },
};

fn extract_text(value: &AttributeValue) -> Option<Text> {
    match value {
        AttributeValue::Text(text) => Some(text.clone()),
        _ => None,
    }
}

fn extract_f32(value: &AttributeValue) -> Option<f32> {
    match value {
        AttributeValue::F32(f32) => Some(f32.clone()),
        _ => None,
    }
}

fn extract_vec2_f32(value: &AttributeValue) -> Option<Vec2<f32>> {
    match value {
        AttributeValue::FloatVec2(vec) => Some(vec.clone()),
        _ => None,
    }
}

fn extract_vec2_i32(value: &AttributeValue) -> Option<Vec2<i32>> {
    match value {
        AttributeValue::IntVec2(vec) => Some(vec.clone()),
        _ => None,
    }
}

fn extract_matrix4x4(value: &AttributeValue) -> Option<Matrix4x4> {
    match value {
        AttributeValue::Matrix4x4(mat) => Some(mat.clone()),
        _ => None,
    }
}

fn extract_rational(value: &AttributeValue) -> Option<Rational> {
    match value {
        AttributeValue::Rational(rat) => Some(rat.clone()),
        _ => None,
    }
}

fn extract_environment_map(value: &AttributeValue) -> Option<EnvironmentMap> {
    match value {
        AttributeValue::EnvironmentMap(env) => Some(env.clone()),
        _ => None,
    }
}

fn extract_key_code(value: &AttributeValue) -> Option<KeyCode> {
    match value {
        AttributeValue::KeyCode(code) => Some(code.clone()),
        _ => None,
    }
}

fn extract_preview(value: &AttributeValue) -> Option<Preview> {
    match value {
        AttributeValue::Preview(preview) => Some(preview.clone()),
        _ => None,
    }
}

fn extract_integer_bounds(value: &AttributeValue) -> Option<IntegerBounds> {
    match value {
        AttributeValue::IntegerBounds(bounds) => Some(bounds.clone()),
        _ => None,
    }
}

fn extract_chromaticities(value: &AttributeValue) -> Option<Chromaticities> {
    match value {
        AttributeValue::Chromaticities(chrom) => Some(chrom.clone()),
        _ => None,
    }
}

fn extract_time_code(value: &AttributeValue) -> Option<TimeCode> {
    match value {
        AttributeValue::TimeCode(tc) => Some(tc.clone()),
        _ => None,
    }
}

struct LayerAttributeHandler<T> {
    name: &'static str,
    extract: fn(&AttributeValue) -> Option<T>,
    get: fn(&LayerAttributes) -> Option<AttributeValue>,
    set: fn(&mut LayerAttributes, T) -> Result<()>,
}

const FLOAT_LAYER_ATTRIBUTES: &[LayerAttributeHandler<f32>] = &[
    LayerAttributeHandler {
        name: "screen_window_width",
        extract: extract_f32,
        get: |attrs| Some(AttributeValue::F32(attrs.screen_window_width.clone())),
        set: |attrs, value| {
            attrs.screen_window_width = value;
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "white_luminance",
        extract: extract_f32,
        get: |attrs| attrs.white_luminance.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.white_luminance = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "horizontal_density",
        extract: extract_f32,
        get: |attrs| attrs.horizontal_density.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.horizontal_density = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "utc_offset",
        extract: extract_f32,
        get: |attrs| attrs.utc_offset.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.utc_offset = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "longitude",
        extract: extract_f32,
        get: |attrs| attrs.longitude.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.longitude = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "latitude",
        extract: extract_f32,
        get: |attrs| attrs.latitude.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.latitude = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "altitude",
        extract: extract_f32,
        get: |attrs| attrs.altitude.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.altitude = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "focus",
        extract: extract_f32,
        get: |attrs| attrs.focus.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.focus = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "exposure",
        extract: extract_f32,
        get: |attrs| attrs.exposure.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.exposure = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "aperture",
        extract: extract_f32,
        get: |attrs| attrs.aperture.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.aperture = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "iso_speed",
        extract: extract_f32,
        get: |attrs| attrs.iso_speed.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.iso_speed = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "near_clip_plane",
        extract: extract_f32,
        get: |attrs| attrs.near_clip_plane.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.near_clip_plane = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "far_clip_plane",
        extract: extract_f32,
        get: |attrs| attrs.far_clip_plane.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.far_clip_plane = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "horizontal_field_of_view",
        extract: extract_f32,
        get: |attrs| attrs.horizontal_field_of_view.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.horizontal_field_of_view = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "vertical_field_of_view",
        extract: extract_f32,
        get: |attrs| attrs.vertical_field_of_view.map(AttributeValue::F32),
        set: |attrs, value| {
            attrs.vertical_field_of_view = Some(value);
            Ok(())
        },
    },
];

const VEC2_F32_LAYER_ATTRIBUTES: &[LayerAttributeHandler<Vec2<f32>>] = &[
    LayerAttributeHandler {
        name: "screen_window_center",
        extract: extract_vec2_f32,
        get: |attrs| {
            Some(AttributeValue::FloatVec2(
                attrs.screen_window_center.clone(),
            ))
        },
        set: |attrs, value| {
            attrs.screen_window_center = value;
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "adopted_neutral",
        extract: extract_vec2_f32,
        get: |attrs| attrs.adopted_neutral.map(AttributeValue::FloatVec2),
        set: |attrs, value| {
            attrs.adopted_neutral = Some(value);
            Ok(())
        },
    },
];

const VEC2_I32_LAYER_ATTRIBUTES: &[LayerAttributeHandler<Vec2<i32>>] = &[LayerAttributeHandler {
    name: "layer_position",
    extract: extract_vec2_i32,
    get: |attrs| Some(AttributeValue::IntVec2(attrs.layer_position.clone())),
    set: |attrs, value| {
        attrs.layer_position = value;
        Ok(())
    },
}];

fn text2attr(text: Option<&Text>) -> Option<AttributeValue> {
    text.map(|text| AttributeValue::Text(text.clone()))
}

const TEXT_LAYER_ATTRIBUTES: &[LayerAttributeHandler<Text>] = &[
    LayerAttributeHandler {
        name: "layer_name",
        extract: extract_text,
        get: |attrs| text2attr(attrs.layer_name.as_ref()),
        set: |attrs, value| {
            attrs.layer_name = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "rendering_transform_name",
        extract: extract_text,
        get: |attrs| text2attr(attrs.rendering_transform_name.as_ref()),
        set: |attrs, value| {
            attrs.rendering_transform_name = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "look_modification_transform_name",
        extract: extract_text,
        get: |attrs| text2attr(attrs.look_modification_transform_name.as_ref()),
        set: |attrs, value| {
            attrs.look_modification_transform_name = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "owner",
        extract: extract_text,
        get: |attrs| text2attr(attrs.owner.as_ref()),
        set: |attrs, value| {
            attrs.owner = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "comments",
        extract: extract_text,
        get: |attrs| text2attr(attrs.comments.as_ref()),
        set: |attrs, value| {
            attrs.comments = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "capture_date",
        extract: extract_text,
        get: |attrs| text2attr(attrs.capture_date.as_ref()),
        set: |attrs, value| {
            attrs.capture_date = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "wrap_mode_name",
        extract: extract_text,
        get: |attrs| text2attr(attrs.wrap_mode_name.as_ref()),
        set: |attrs, value| {
            attrs.wrap_mode_name = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "view_name",
        extract: extract_text,
        get: |attrs| text2attr(attrs.view_name.as_ref()),
        set: |attrs, value| {
            attrs.view_name = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "software_name",
        extract: extract_text,
        get: |attrs| text2attr(attrs.software_name.as_ref()),
        set: |attrs, value| {
            attrs.software_name = Some(value);
            Ok(())
        },
    },
];

const MATRIX4X4_LAYER_ATTRIBUTES: &[LayerAttributeHandler<Matrix4x4>] = &[
    LayerAttributeHandler {
        name: "world_to_camera",
        extract: extract_matrix4x4,
        get: |attrs| attrs.world_to_camera.map(AttributeValue::Matrix4x4),
        set: |attrs, value| {
            attrs.world_to_camera = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "world_to_normalized_device",
        extract: extract_matrix4x4,
        get: |attrs| {
            attrs
                .world_to_normalized_device
                .map(AttributeValue::Matrix4x4)
        },
        set: |attrs, value| {
            attrs.world_to_normalized_device = Some(value);
            Ok(())
        },
    },
];

const ENVIRONMENT_MAP_LAYER_ATTRIBUTES: &[LayerAttributeHandler<EnvironmentMap>] =
    &[LayerAttributeHandler {
        name: "environment_map",
        extract: extract_environment_map,
        get: |attrs| attrs.environment_map.map(AttributeValue::EnvironmentMap),
        set: |attrs, value| {
            attrs.environment_map = Some(value);
            Ok(())
        },
    }];

const KEY_CODE_LAYER_ATTRIBUTES: &[LayerAttributeHandler<KeyCode>] = &[LayerAttributeHandler {
    name: "film_key_code",
    extract: extract_key_code,
    get: |attrs| attrs.film_key_code.map(AttributeValue::KeyCode),
    set: |attrs, value| {
        attrs.film_key_code = Some(value);
        Ok(())
    },
}];

const RATIONAL_LAYER_ATTRIBUTES: &[LayerAttributeHandler<Rational>] = &[
    LayerAttributeHandler {
        name: "frames_per_second",
        extract: extract_rational,
        get: |attrs| attrs.frames_per_second.map(AttributeValue::Rational),
        set: |attrs, value| {
            attrs.frames_per_second = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "deep_image_state",
        extract: extract_rational,
        get: |attrs| attrs.deep_image_state.map(AttributeValue::Rational),
        set: |attrs, value| {
            attrs.deep_image_state = Some(value);
            Ok(())
        },
    },
];

const INTEGER_BOUNDS_LAYER_ATTRIBUTES: &[LayerAttributeHandler<IntegerBounds>] =
    &[LayerAttributeHandler {
        name: "original_data_window",
        extract: extract_integer_bounds,
        get: |attrs| {
            attrs
                .original_data_window
                .map(AttributeValue::IntegerBounds)
        },
        set: |attrs, value| {
            attrs.original_data_window = Some(value);
            Ok(())
        },
    }];

pub fn attributes_from_layer(layer_attributes: &LayerAttributes) -> HashMap<Text, AttributeValue> {
    let mut attributes = layer_attributes.other.clone();

    for handler in FLOAT_LAYER_ATTRIBUTES {
        if let Some(value) = (handler.get)(layer_attributes) {
            attributes.insert(Text::from(handler.name), value);
        }
    }

    for handler in VEC2_F32_LAYER_ATTRIBUTES {
        if let Some(value) = (handler.get)(layer_attributes) {
            attributes.insert(Text::from(handler.name), value);
        }
    }

    for handler in VEC2_I32_LAYER_ATTRIBUTES {
        if let Some(value) = (handler.get)(layer_attributes) {
            attributes.insert(Text::from(handler.name), value);
        }
    }

    for handler in TEXT_LAYER_ATTRIBUTES {
        if let Some(value) = (handler.get)(layer_attributes) {
            attributes.insert(Text::from(handler.name), value);
        }
    }

    for handler in MATRIX4X4_LAYER_ATTRIBUTES {
        if let Some(value) = (handler.get)(layer_attributes) {
            attributes.insert(Text::from(handler.name), value);
        }
    }

    for handler in ENVIRONMENT_MAP_LAYER_ATTRIBUTES {
        if let Some(value) = (handler.get)(layer_attributes) {
            attributes.insert(Text::from(handler.name), value);
        }
    }

    for handler in KEY_CODE_LAYER_ATTRIBUTES {
        if let Some(value) = (handler.get)(layer_attributes) {
            attributes.insert(Text::from(handler.name), value);
        }
    }

    for handler in RATIONAL_LAYER_ATTRIBUTES {
        if let Some(value) = (handler.get)(layer_attributes) {
            attributes.insert(Text::from(handler.name), value);
        }
    }

    for handler in INTEGER_BOUNDS_LAYER_ATTRIBUTES {
        if let Some(value) = (handler.get)(layer_attributes) {
            attributes.insert(Text::from(handler.name), value);
        }
    }

    attributes
}

pub fn layer_attributes_from_attributes(
    layer_attributes: &mut LayerAttributes,
    attributes: &HashMap<Text, AttributeValue>,
) -> Result<()> {
    let mut attributes = attributes.clone();

    for handler in FLOAT_LAYER_ATTRIBUTES {
        let extracted_value = attributes
            .remove(&Text::from(handler.name))
            .map(|value| (handler.extract)(&value))
            .flatten();

        if let Some(value) = extracted_value {
            match (handler.set)(layer_attributes, value) {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
    }

    for handler in VEC2_F32_LAYER_ATTRIBUTES {
        let extracted_value = attributes
            .remove(&Text::from(handler.name))
            .map(|value| (handler.extract)(&value))
            .flatten();

        if let Some(value) = extracted_value {
            match (handler.set)(layer_attributes, value) {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
    }

    for handler in VEC2_I32_LAYER_ATTRIBUTES {
        let extracted_value = attributes
            .remove(&Text::from(handler.name))
            .map(|value| (handler.extract)(&value))
            .flatten();

        if let Some(value) = extracted_value {
            match (handler.set)(layer_attributes, value) {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
    }

    for handler in TEXT_LAYER_ATTRIBUTES {
        let extracted_value = attributes
            .remove(&Text::from(handler.name))
            .map(|value| (handler.extract)(&value))
            .flatten();

        if let Some(value) = extracted_value {
            match (handler.set)(layer_attributes, value) {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
    }

    for handler in MATRIX4X4_LAYER_ATTRIBUTES {
        let extracted_value = attributes
            .remove(&Text::from(handler.name))
            .map(|value| (handler.extract)(&value))
            .flatten();

        if let Some(value) = extracted_value {
            match (handler.set)(layer_attributes, value) {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
    }

    for handler in ENVIRONMENT_MAP_LAYER_ATTRIBUTES {
        let extracted_value = attributes
            .remove(&Text::from(handler.name))
            .map(|value| (handler.extract)(&value))
            .flatten();

        if let Some(value) = extracted_value {
            match (handler.set)(layer_attributes, value) {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
    }

    for handler in KEY_CODE_LAYER_ATTRIBUTES {
        let extracted_value = attributes
            .remove(&Text::from(handler.name))
            .map(|value| (handler.extract)(&value))
            .flatten();

        if let Some(value) = extracted_value {
            match (handler.set)(layer_attributes, value) {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
    }

    for handler in RATIONAL_LAYER_ATTRIBUTES {
        let extracted_value = attributes
            .remove(&Text::from(handler.name))
            .map(|value| (handler.extract)(&value))
            .flatten();

        if let Some(value) = extracted_value {
            match (handler.set)(layer_attributes, value) {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
    }

    for handler in INTEGER_BOUNDS_LAYER_ATTRIBUTES {
        let extracted_value = attributes
            .remove(&Text::from(handler.name))
            .map(|value| (handler.extract)(&value))
            .flatten();

        if let Some(value) = extracted_value {
            match (handler.set)(layer_attributes, value) {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }
    }

    Ok(())
}

struct ImageAttributeHandler {
    name: &'static str,
    get: fn(&ImageAttributes) -> Option<AttributeValue>,
    set: fn(&mut ImageAttributes, AttributeValue) -> Result<()>,
}

const IMAGE_ATTRIBUTES: &[ImageAttributeHandler] = &[
    ImageAttributeHandler {
        name: "display_window",
        get: |attrs| Some(AttributeValue::IntegerBounds(attrs.display_window.clone())),
        set: |attrs, value| {
            if let AttributeValue::IntegerBounds(bounds) = value {
                attrs.display_window = bounds;
            }

            Ok(())
        },
    },
    ImageAttributeHandler {
        name: "pixel_aspect_ratio",
        get: |attrs| Some(AttributeValue::F32(attrs.pixel_aspect.clone())),
        set: |attrs, value| {
            if let AttributeValue::F32(pixel_aspect) = value {
                attrs.pixel_aspect = pixel_aspect;
            }

            Ok(())
        },
    },
    ImageAttributeHandler {
        name: "chromaticities",
        get: |attrs| {
            attrs
                .chromaticities
                .as_ref()
                .map(|c| AttributeValue::Chromaticities(c.clone()))
        },
        set: |attrs, value| {
            if let AttributeValue::Chromaticities(chrom) = value {
                attrs.chromaticities = Some(chrom);
            }
            Ok(())
        },
    },
    ImageAttributeHandler {
        name: "time_code",
        get: |attrs| {
            attrs
                .time_code
                .as_ref()
                .map(|tc| AttributeValue::TimeCode(tc.clone()))
        },
        set: |attrs, value| {
            if let AttributeValue::TimeCode(tc) = value {
                attrs.time_code = Some(tc);
            }
            Ok(())
        },
    },
];

pub fn attributes_from_image(attributes: &ImageAttributes) -> HashMap<Text, AttributeValue> {
    let mut image_attributes = attributes.other.clone();

    for handler in IMAGE_ATTRIBUTES {
        if let Some(value) = (handler.get)(attributes) {
            image_attributes.insert(Text::from(handler.name), value);
        }
    }

    return image_attributes;
}

pub fn image_attributes_from_attributes(
    image_attributes: &mut ImageAttributes,
    _attributes: &HashMap<Text, AttributeValue>,
) -> Result<()> {
    let mut attributes = _attributes.clone();

    for handler in IMAGE_ATTRIBUTES {
        match attributes.remove(&Text::from(handler.name)) {
            Some(value) => match (handler.set)(image_attributes, value) {
                Ok(_) => (),
                Err(e) => return Err(e),
            },
            None => (),
        }
    }

    image_attributes.other = attributes;

    Ok(())
}
