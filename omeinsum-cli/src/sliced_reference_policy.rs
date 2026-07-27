#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SlicedReferencePolicy {
    Compute,
    OmittedUserAuthorizedNpuFirst,
}

impl SlicedReferencePolicy {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value {
            "compute" => Ok(Self::Compute),
            "omitted-user-authorized-npu-first" => Ok(Self::OmittedUserAuthorizedNpuFirst),
            _ => Err(format!(
                "unsupported sliced reference policy {value:?}; expected compute or \
                 omitted-user-authorized-npu-first"
            )),
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::OmittedUserAuthorizedNpuFirst => "omitted-user-authorized-npu-first",
        }
    }

    pub(crate) fn correctness_admission(self) -> &'static str {
        match self {
            Self::Compute => "reference-computed",
            Self::OmittedUserAuthorizedNpuFirst => "not-performed",
        }
    }

    pub(crate) fn evaluate<T>(
        self,
        backend: &str,
        compute: impl FnOnce() -> Result<T, String>,
    ) -> Result<Option<T>, String> {
        match self {
            Self::Compute => compute().map(Some),
            Self::OmittedUserAuthorizedNpuFirst if backend == "ascend" => Ok(None),
            Self::OmittedUserAuthorizedNpuFirst => Err(
                "--reference-policy omitted-user-authorized-npu-first is only valid with \
                 --backend ascend"
                    .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::SlicedReferencePolicy;

    #[test]
    fn omitted_npu_first_policy_does_not_compute_the_reference() {
        let called = Cell::new(false);
        let result = SlicedReferencePolicy::OmittedUserAuthorizedNpuFirst
            .evaluate("ascend", || {
                called.set(true);
                Ok::<_, String>(41)
            })
            .unwrap();

        assert_eq!(result, None);
        assert!(!called.get());
    }

    #[test]
    fn omitted_npu_first_policy_is_rejected_on_cpu_without_computing() {
        let called = Cell::new(false);
        let error = SlicedReferencePolicy::OmittedUserAuthorizedNpuFirst
            .evaluate("cpu", || {
                called.set(true);
                Ok::<_, String>(41)
            })
            .unwrap_err();

        assert!(error.contains("only valid with --backend ascend"));
        assert!(!called.get());
    }

    #[test]
    fn compute_policy_preserves_the_reference() {
        let result = SlicedReferencePolicy::Compute
            .evaluate("cpu", || Ok::<_, String>(41))
            .unwrap();

        assert_eq!(result, Some(41));
    }
}
