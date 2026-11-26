//! Pure-Rust algorithms behind Spatex. No Rustler types here, so everything is unit-testable
//! and reusable outside Elixir.

pub mod classify;
pub mod contiguity;
pub mod dynamics;
pub mod esda;
pub mod graph;
pub mod gwr;
pub mod inequality;
pub mod limits;
pub mod perm;
pub mod pointpats;
pub mod region;
pub mod regression;
pub mod weights;

/// Version of the Rust-side ABI the Elixir layer expects (bumped on breaking NIF changes).
pub fn abi_version() -> u32 {
    1
}

/// Decodes a native-endian `f64` byte slice (the NIF wire format for numeric vectors).
pub fn bytes_to_f64s(bytes: &[u8]) -> Result<Vec<f64>, String> {
    if !bytes.len().is_multiple_of(8) {
        return Err(format!(
            "f64 binary length {} is not a multiple of 8",
            bytes.len()
        ));
    }
    let (chunks, _) = bytes.as_chunks::<8>();
    Ok(chunks.iter().map(|c| f64::from_ne_bytes(*c)).collect())
}

/// Decodes a native-endian `u32` byte slice (the NIF wire format for index vectors).
pub fn bytes_to_u32s(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "u32 binary length {} is not a multiple of 4",
            bytes.len()
        ));
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    Ok(chunks.iter().map(|c| u32::from_ne_bytes(*c)).collect())
}

/// Sums a slice of `f64`.
pub fn sum_f64(values: &[f64]) -> f64 {
    values.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_version_is_one() {
        assert_eq!(abi_version(), 1);
    }

    #[test]
    fn bytes_to_f64s_roundtrips_native_endian() {
        let values = [1.5_f64, -2.0, f64::INFINITY, 0.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        assert_eq!(bytes_to_f64s(&bytes).unwrap(), values);
    }

    #[test]
    fn bytes_to_f64s_accepts_empty() {
        assert_eq!(bytes_to_f64s(&[]).unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn bytes_to_f64s_rejects_bad_length() {
        let err = bytes_to_f64s(&[0u8; 9]).unwrap_err();
        assert!(err.contains("multiple of 8"), "{err}");
    }

    #[test]
    fn bytes_to_u32s_roundtrips_and_rejects_bad_length() {
        let values = [0_u32, 7, u32::MAX];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        assert_eq!(bytes_to_u32s(&bytes).unwrap(), values);
        assert!(bytes_to_u32s(&[0u8; 5])
            .unwrap_err()
            .contains("multiple of 4"));
    }

    #[test]
    fn sum_f64_adds() {
        assert_eq!(sum_f64(&[1.0, 2.5, -0.5]), 3.0);
        assert_eq!(sum_f64(&[]), 0.0);
    }
}
