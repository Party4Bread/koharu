use std::sync::Arc;

use crate::{
    Asset, AssetRole, BlobId, Edit, EntityId, Error, FunctionalRelation, Patch, ProjectId,
    RelationId, RelationKind, RelationSpec, Result,
    component::{Component, ValidationContext, decode, key},
    components::Assets,
    state::State,
};

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub(crate) state: Arc<State>,
    pub(crate) storage: koharu_storage::State,
}

impl Snapshot {
    pub(crate) fn new(state: Arc<State>, storage: koharu_storage::State) -> Result<Self> {
        if state.document != storage.document_id() || state.revision != storage.revision() {
            return Err(Error::invalid(
                "scene state and blob snapshot identify different revisions",
            ));
        }
        Ok(Self { state, storage })
    }

    #[must_use]
    pub fn project_id(&self) -> ProjectId {
        ProjectId(self.state.document)
    }

    #[must_use]
    pub fn revision(&self) -> crate::Revision {
        self.state.revision
    }

    pub fn pages(&self) -> impl ExactSizeIterator<Item = PageRef<'_>> {
        self.state
            .page_order
            .iter()
            .copied()
            .map(|id| PageRef { snapshot: self, id })
    }

    pub fn entities(&self) -> impl ExactSizeIterator<Item = EntityRef<'_>> {
        let ids = self
            .state
            .page_order
            .iter()
            .flat_map(|page| self.state.pages[page].ordered_ids())
            .collect::<Vec<_>>();
        ids.into_iter().map(|id| EntityRef { snapshot: self, id })
    }

    pub fn subtree(&self, root: EntityId) -> Result<impl Iterator<Item = EntityRef<'_>> + '_> {
        let ids = self.state.page_containing(root)?.descendants(root)?;
        Ok(ids.into_iter().map(|id| EntityRef { snapshot: self, id }))
    }

    pub fn descendants(&self, root: EntityId) -> Result<impl Iterator<Item = EntityRef<'_>> + '_> {
        Ok(self.subtree(root)?.filter(move |entity| entity.id != root))
    }

    pub fn entities_with<T: Component>(
        &self,
    ) -> Result<impl ExactSizeIterator<Item = EntityRef<'_>>> {
        let key = key::<T>()?;
        let ids = self
            .state
            .page_order
            .iter()
            .flat_map(|page| self.state.pages[page].entities_with(&key))
            .collect::<Vec<_>>();
        Ok(ids.into_iter().map(|id| EntityRef { snapshot: self, id }))
    }

    pub fn page(&self, id: EntityId) -> Result<PageRef<'_>> {
        if self.state.pages.contains_key(&id) {
            Ok(PageRef { snapshot: self, id })
        } else {
            Err(Error::EntityNotFound(id))
        }
    }

    pub fn entity(&self, id: EntityId) -> Result<EntityRef<'_>> {
        if self.state.contains_entity(id) {
            Ok(EntityRef { snapshot: self, id })
        } else {
            Err(Error::EntityNotFound(id))
        }
    }

    pub fn relation(&self, id: RelationId) -> Result<RelationRef<'_>> {
        if self.state.relations.contains_key(&id) {
            Ok(RelationRef { snapshot: self, id })
        } else {
            Err(Error::RelationNotFound(id))
        }
    }

    pub fn parent(&self, id: EntityId) -> Result<Option<EntityId>> {
        let page_id = self.state.page_for(id)?;
        if id == page_id {
            return Ok(None);
        }
        let page = self.state.page(page_id)?;
        let key = page.key(id)?;
        Ok(page.entities[key]
            .parent
            .map(|parent| page.entities[parent].id))
    }

    pub fn children(&self, id: EntityId) -> Result<impl ExactSizeIterator<Item = EntityId> + '_> {
        let page = self.state.page_containing(id)?;
        let entity = page.entity(id)?;
        Ok(entity.children.iter().map(|child| page.entities[*child].id))
    }

    pub fn component<T: Component>(&self, entity: EntityId) -> Result<Option<T>> {
        let key = key::<T>()?;
        self.decode(self.state.component(entity, &key)?)
    }

    pub fn project_component<T: Component>(&self) -> Result<Option<T>> {
        let key = key::<T>()?;
        self.decode(self.state.project_component(&key))
    }

    pub fn asset(&self, entity: EntityId, role: &AssetRole) -> Result<Option<Asset>> {
        Ok(self
            .component::<Assets>(entity)?
            .and_then(|assets| assets.values.get(role).cloned()))
    }

    fn relation_component<T: Component>(&self, relation: RelationId) -> Result<Option<T>> {
        let key = key::<T>()?;
        self.decode(self.state.relation_component(relation, &key)?)
    }

    fn decode<T: Component>(
        &self,
        raw: Option<&crate::component::ComponentRecord>,
    ) -> Result<Option<T>> {
        let Some(raw) = raw else {
            return Ok(None);
        };
        let record_exists = |id| self.state.contains_entity(id);
        let blob_exists = |id| self.storage.blobs().contains(id);
        decode::<T>(raw, &ValidationContext::new(&record_exists, &blob_exists)).map(Some)
    }

    pub fn relations_from<'a>(
        &'a self,
        entity: EntityId,
        kind: Option<&'a RelationKind>,
    ) -> impl Iterator<Item = RelationRef<'a>> + 'a {
        self.state
            .outgoing
            .get(&entity)
            .into_iter()
            .flat_map(|ids| ids.iter().copied())
            .filter(move |id| kind.is_none_or(|kind| self.state.relations[id].value.kind == *kind))
            .map(|id| RelationRef { snapshot: self, id })
    }

    pub fn relations_to<'a>(
        &'a self,
        entity: EntityId,
        kind: Option<&'a RelationKind>,
    ) -> impl Iterator<Item = RelationRef<'a>> + 'a {
        self.state
            .incoming
            .get(&entity)
            .into_iter()
            .flat_map(|ids| ids.iter().copied())
            .filter(move |id| kind.is_none_or(|kind| self.state.relations[id].value.kind == *kind))
            .map(|id| RelationRef { snapshot: self, id })
    }

    pub fn relations_from_as<R: RelationSpec>(
        &self,
        entity: EntityId,
    ) -> impl Iterator<Item = RelationRef<'_>> {
        let kind = R::kind();
        self.relations_from(entity, None)
            .filter(move |relation| relation.value().kind == kind)
    }

    pub fn relations_to_as<R: RelationSpec>(
        &self,
        entity: EntityId,
    ) -> impl Iterator<Item = RelationRef<'_>> {
        let kind = R::kind();
        self.relations_to(entity, None)
            .filter(move |relation| relation.value().kind == kind)
    }

    pub fn relation_from<R: FunctionalRelation>(
        &self,
        entity: EntityId,
    ) -> Result<Option<RelationRef<'_>>> {
        let mut relations = self.relations_from_as::<R>(entity);
        let relation = relations.next();
        if relations.next().is_some() {
            return Err(Error::invalid(format!(
                "entity {entity} has multiple {} relations",
                R::KIND
            )));
        }
        Ok(relation)
    }

    pub async fn read_blob(&self, id: BlobId) -> Result<bytes::Bytes> {
        self.storage.blobs().get(id).await.map_err(Into::into)
    }

    /// Fingerprints the complete content and hierarchy owned by one page, including incident
    /// relations. Blob identifiers are content hashes, so referenced raster inputs participate
    /// without reading their bytes again.
    pub fn page_content_fingerprint(&self, id: EntityId) -> Result<[u8; 32]> {
        let page = self.state.page(id)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"koharu-scene-page-content-v1\0");
        for entity_id in page.ordered_ids() {
            let entity = page.entity(entity_id)?;
            hash_fingerprint_bytes(&mut hasher, entity_id.as_uuid().as_bytes());
            match entity.parent {
                Some(parent) => {
                    hasher.update(&[1]);
                    hash_fingerprint_bytes(
                        &mut hasher,
                        page.entities[parent].id.as_uuid().as_bytes(),
                    );
                }
                None => {
                    hasher.update(&[0]);
                }
            }
            for (key, component) in &entity.components {
                hash_fingerprint_bytes(&mut hasher, key.kind.as_bytes());
                hash_fingerprint_bytes(&mut hasher, &component.fingerprint());
            }
            hasher.update(&[0xff]);
        }
        let mut relations = self
            .state
            .relations
            .iter()
            .filter(|(_, relation)| {
                self.state.entity_pages.get(&relation.value.source) == Some(&id)
                    || self.state.entity_pages.get(&relation.value.target) == Some(&id)
            })
            .collect::<Vec<_>>();
        relations.sort_unstable_by_key(|(relation_id, _)| **relation_id);
        for (relation_id, relation) in relations {
            hash_fingerprint_bytes(&mut hasher, relation_id.as_uuid().as_bytes());
            hash_fingerprint_bytes(&mut hasher, &revision::to_vec(&relation.value)?);
            for (key, component) in &relation.components {
                hash_fingerprint_bytes(&mut hasher, key.kind.as_bytes());
                hash_fingerprint_bytes(&mut hasher, &component.fingerprint());
            }
            hasher.update(&[0xfe]);
        }
        Ok(*hasher.finalize().as_bytes())
    }

    #[must_use]
    pub fn edit(&self) -> Edit {
        Edit::new(self.clone(), None)
    }

    #[must_use]
    pub fn edit_as(&self, generation: crate::Generation) -> Edit {
        Edit::new(self.clone(), Some(generation))
    }

    pub fn patch(&self, f: impl FnOnce(&mut Edit) -> Result<()>) -> Result<Patch> {
        let mut edit = self.edit();
        f(&mut edit)?;
        edit.finish()
    }

    pub fn preview<'a>(&self, patches: impl IntoIterator<Item = &'a Patch>) -> Result<Self> {
        let mut current = self.clone();
        for patch in patches {
            let patch = patch.rebase_on(&current)?;
            let mut state = (*patch.state).clone();
            state.revision = current.state.revision;
            let state = Arc::new(state);
            let storage = current.storage.update(
                state.revision,
                current.storage.payload().clone(),
                state.referenced_blobs(),
                patch.attachments.iter().cloned(),
            )?;
            current = Self { state, storage };
        }
        Ok(current)
    }
}

fn hash_fingerprint_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Copy, Clone, Debug)]
pub struct EntityRef<'snapshot> {
    snapshot: &'snapshot Snapshot,
    id: EntityId,
}

impl<'snapshot> EntityRef<'snapshot> {
    #[must_use]
    pub const fn id(self) -> EntityId {
        self.id
    }

    pub fn component<T: Component>(self) -> Result<Option<T>> {
        self.snapshot.component(self.id)
    }

    pub fn parent(self) -> Result<Option<EntityId>> {
        self.snapshot.parent(self.id)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PageRef<'snapshot> {
    pub(crate) snapshot: &'snapshot Snapshot,
    pub(crate) id: EntityId,
}

impl<'snapshot> PageRef<'snapshot> {
    #[must_use]
    pub const fn id(self) -> EntityId {
        self.id
    }

    pub fn page(self) -> Result<crate::Page> {
        self.snapshot
            .component(self.id)?
            .ok_or_else(|| Error::invalid("page component is missing"))
    }

    #[must_use]
    pub fn entity(self) -> EntityRef<'snapshot> {
        EntityRef {
            snapshot: self.snapshot,
            id: self.id,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct RelationRef<'snapshot> {
    pub(crate) snapshot: &'snapshot Snapshot,
    pub(crate) id: RelationId,
}

impl<'snapshot> RelationRef<'snapshot> {
    #[must_use]
    pub const fn id(self) -> RelationId {
        self.id
    }

    #[must_use]
    pub fn value(self) -> &'snapshot crate::Relation {
        &self.snapshot.state.relations[&self.id].value
    }

    pub fn component<T: Component>(self) -> Result<Option<T>> {
        self.snapshot.relation_component(self.id)
    }
}
