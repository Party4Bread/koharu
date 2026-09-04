use keyring_core::{CredentialStore, Entry, Error};
use secrecy::{SecretBox, zeroize::Zeroize};
use serde::{Deserialize, Serialize, Serializer};
use std::sync::{Arc, LazyLock};

pub use secrecy::{ExposeSecret, SerializableSecret};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(transparent)]
pub struct SecretString(SecretBox<SecretValue>);

impl Default for SecretString {
    fn default() -> Self {
        Self::from(String::new())
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(SecretBox::new(Box::new(SecretValue(value))))
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self::from(value.to_owned())
    }
}

impl ExposeSecret<str> for SecretString {
    fn expose_secret(&self) -> &str {
        &self.0.expose_secret().0
    }
}

#[derive(Clone, Deserialize)]
#[serde(transparent)]
struct SecretValue(String);

impl Zeroize for SecretValue {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

impl secrecy::CloneableSecret for SecretValue {}

impl Serialize for SecretValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str("[REDACTED]")
    }
}

impl SerializableSecret for SecretValue {}

const SERVICE: &str = "koharu";

static CREDENTIAL_STORE: LazyLock<Result<Arc<CredentialStore>, String>> = LazyLock::new(|| {
    #[cfg(test)]
    let store: Result<Arc<CredentialStore>, Error> =
        keyring_core::mock::Store::new().map(|store| store as _);
    #[cfg(all(not(test), target_os = "linux"))]
    let store: Result<Arc<CredentialStore>, Error> =
        zbus_secret_service_keyring_store::Store::new().map(|store| store as _);
    #[cfg(all(not(test), windows))]
    let store: Result<Arc<CredentialStore>, Error> =
        windows_native_keyring_store::Store::new().map(|store| store as _);
    #[cfg(all(not(test), target_os = "macos"))]
    let store: Result<Arc<CredentialStore>, Error> =
        apple_native_keyring_store::keychain::Store::new().map(|store| store as _);
    #[cfg(all(not(test), not(any(target_os = "linux", windows, target_os = "macos"))))]
    let store = Err(Error::Invalid(
        "platform".to_owned(),
        "Koharu secrets require Linux, Windows, or macOS".to_owned(),
    ));
    store.map_err(|error| error.to_string())
});

/// Load a Koharu secret by key, returning `None` when no credential exists.
pub fn get(key: &str) -> anyhow::Result<Option<SecretString>> {
    let entry = entry(key)?;
    get_from_entry(&entry)
}

fn get_from_entry(entry: &Entry) -> anyhow::Result<Option<SecretString>> {
    match entry.get_password() {
        Ok(value) => Ok(Some(SecretString::from(value))),
        Err(Error::NoEntry) => Ok(None),
        Err(error) => Err(operation_error("read a credential", error)),
    }
}

/// Store a Koharu secret by key.
pub fn set(key: &str, secret: &SecretString) -> anyhow::Result<()> {
    let entry = entry(key)?;
    set_on_entry(&entry, secret)
}

fn set_on_entry(entry: &Entry, secret: &SecretString) -> anyhow::Result<()> {
    entry
        .set_password(secret.expose_secret())
        .map_err(|error| operation_error("store a credential", error))
}

/// Delete a Koharu secret by key. Missing credentials are treated as success.
pub fn delete(key: &str) -> anyhow::Result<()> {
    let entry = entry(key)?;
    delete_from_entry(&entry)
}

fn delete_from_entry(entry: &Entry) -> anyhow::Result<()> {
    match entry.delete_credential() {
        Ok(()) | Err(Error::NoEntry) => Ok(()),
        Err(error) => Err(operation_error("delete a credential", error)),
    }
}

fn entry(key: &str) -> anyhow::Result<Entry> {
    let store = CREDENTIAL_STORE
        .as_ref()
        .map_err(|error| initialization_error(error))?;
    store
        .build(SERVICE, key, None)
        .map_err(|error| operation_error("access a credential", error))
}

fn initialization_error(error: &str) -> anyhow::Error {
    #[cfg(target_os = "linux")]
    return anyhow::anyhow!(
        "Linux Secret Service is unavailable. Ensure a Secret Service provider (such as GNOME Keyring or KWallet's Secret Service interface) is running and the login collection is unlocked: {error}"
    );
    #[cfg(not(target_os = "linux"))]
    anyhow::anyhow!("failed to initialize the operating system credential store: {error}")
}

fn operation_error(operation: &str, error: Error) -> anyhow::Error {
    #[cfg(target_os = "linux")]
    {
        let context = if matches!(
            &error,
            Error::NoStorageAccess(_) | Error::PlatformFailure(_)
        ) {
            format!(
                "Linux Secret Service is unavailable or locked while trying to {operation}; ensure a Secret Service provider (such as GNOME Keyring or KWallet's Secret Service interface) is running and the login collection is unlocked"
            )
        } else {
            format!("Linux Secret Service could not {operation}")
        };
        return anyhow::Error::new(error).context(context);
    }
    #[cfg(not(target_os = "linux"))]
    anyhow::Error::new(error).context(format!(
        "the operating system credential store could not {operation}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SECRET: &str = "test credential contents";

    #[test]
    fn secret_api_round_trip_and_delete_are_backend_independent() {
        let secret = SecretString::from(TEST_SECRET);

        assert!(
            get("test-round-trip")
                .expect("missing credential should be readable")
                .is_none()
        );
        delete("test-round-trip").expect("deleting a missing credential should succeed");

        set("test-round-trip", &secret).expect("credential should be stored");
        let stored = get("test-round-trip")
            .expect("stored credential should be readable")
            .expect("stored credential should exist");
        assert!(stored.expose_secret() == secret.expose_secret());

        delete("test-round-trip").expect("stored credential should be deleted");
        assert!(
            get("test-round-trip")
                .expect("deleted credential should be readable as missing")
                .is_none()
        );
    }

    #[test]
    fn replacing_a_secret_does_not_expose_it_in_debug_output() {
        let first = SecretString::from(TEST_SECRET);
        let replacement = SecretString::from("replacement credential contents");

        set("test-replace", &first).expect("initial credential should be stored");
        set("test-replace", &replacement).expect("credential should be replaced");
        let stored = get("test-replace")
            .expect("replacement credential should be readable")
            .expect("replacement credential should exist");

        assert!(stored.expose_secret() == replacement.expose_secret());
        assert!(!format!("{stored:?}").contains(replacement.expose_secret()));
        delete("test-replace").expect("replacement credential should be deleted");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn secret_service_access_errors_are_actionable_and_redacted() {
        let error = Error::NoStorageAccess(Box::new(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "credential collection is locked",
        )));
        let message = format!("{:#}", operation_error("read a credential", error));

        assert!(message.contains("Linux Secret Service is unavailable or locked"));
        assert!(message.contains("login collection is unlocked"));
        assert!(!message.contains(TEST_SECRET));
    }
}
