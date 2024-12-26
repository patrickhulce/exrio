use std::collections::HashMap;

use exr::prelude::{AttributeValue, ImageAttributes, IntegerBounds, LayerAttributes, Result, Text};

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
        name: "utc_offset",
        extract: extract_f32,
        get: |attrs| attrs.utc_offset.map(|value| AttributeValue::F32(value)),
        set: |attrs, value| {
            attrs.utc_offset = Some(value);
            Ok(())
        },
    },
];

const TEXT_LAYER_ATTRIBUTES: &[LayerAttributeHandler<Text>] = &[
    LayerAttributeHandler {
        name: "layer_name",
        extract: extract_text,
        get: |attrs| {
            attrs
                .layer_name
                .as_ref()
                .map(|value| AttributeValue::Text(value.clone()))
        },
        set: |attrs, value| {
            attrs.layer_name = Some(value);
            Ok(())
        },
    },
    LayerAttributeHandler {
        name: "owner",
        extract: extract_text,
        get: |attrs| {
            attrs
                .owner
                .as_ref()
                .map(|value| AttributeValue::Text(value.clone()))
        },
        set: |attrs, value| {
            attrs.layer_name = Some(value);
            Ok(())
        },
    },
];

pub fn attributes_from_layer(layer_attributes: &LayerAttributes) -> HashMap<Text, AttributeValue> {
    let mut attributes = layer_attributes.other.clone();

    for handler in FLOAT_LAYER_ATTRIBUTES {
        if let Some(value) = (handler.get)(layer_attributes) {
            attributes.insert(Text::from(handler.name), value);
        }
    }

    if let Some(layer_name) = &layer_attributes.layer_name {
        attributes.insert(
            Text::from("layer_name"),
            AttributeValue::Text(layer_name.clone()),
        );
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
