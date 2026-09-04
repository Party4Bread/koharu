use crate::acceptance::TextElementInspection;

/// Region identity is structural: both the entity ID and region kind must match the source.
/// Layout-relation evidence does not turn a self-source placement into a container target.
pub(crate) fn is_actual_container_bound(element: &TextElementInspection) -> bool {
    element.text_safe_region.as_ref().is_some_and(|region| {
        Some(region.id) != element.source_region_id
            || Some(region.kind.as_str()) != element.source_region_kind.as_deref()
    })
}
