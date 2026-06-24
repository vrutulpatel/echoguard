# Ethical Guidelines for EchoGuard

## Purpose

EchoGuard is a **detection tool only**. Its sole purpose is to help identify potentially synthetic or manipulated media. It does not generate, improve, or assist in generating deepfakes, voice clones, or any other synthetic media.

---

## Intended Use Cases

EchoGuard is designed for:

- **Journalists and fact-checkers** verifying the authenticity of video or audio before publication.
- **HR and recruiting teams** screening for AI-generated video interview responses.
- **Individuals** protecting themselves against voice fraud and impersonation attempts.
- **Security researchers** studying the detection landscape and advancing the field.
- **Educators** demonstrating how deepfake detection works in academic settings.

---

## Prohibited Uses

EchoGuard must not be used to:

- **Circumvent or improve deepfake generation** — using detection feedback to iteratively improve synthetic media until it evades detection.
- **Mass surveillance** — scanning individuals' media without their knowledge or consent.
- **Discriminatory profiling** — using detection results as grounds for adverse treatment based on protected characteristics.
- **Law enforcement without due process** — using detection output as sole evidence in legal proceedings without independent expert verification and proper legal oversight.
- **Harassment** — weaponizing false positives to defame or discredit individuals.

---

## Limitations and Misuse Risks

Users must understand that:

1. **EchoGuard is not infallible.** It will produce false positives (real content flagged as fake) and false negatives (synthetic content not detected). Detection accuracy depends heavily on available model weights and content type.

2. **A positive detection is not proof.** The verdict "LIKELY DEEPFAKE" is probabilistic, not forensic. It should trigger further investigation, not serve as a standalone conclusion.

3. **Context matters.** Some legitimate content may trigger anomaly flags — e.g., low-quality video, compressed audio, or unusual recording environments. Human judgment is required.

4. **Adversarial deepfakes evolve faster than detectors.** EchoGuard represents a snapshot in the ongoing arms race between generation and detection. Regular model updates are necessary to maintain effectiveness.

---

## Dataset Consent

Any contributions of datasets, training scripts, or model weights must:

- Use data collected with informed consent from the individuals depicted.
- Comply with the licensing terms of the source dataset.
- Not include non-consensual intimate imagery (NCII) or other illegal content.
- Document the data collection methodology and consent process in the PR.

EchoGuard maintainers reserve the right to reject contributions that cannot demonstrate proper consent and licensing.

---

## Responsible Disclosure

If you discover a significant bypass, false-positive attack, or privacy vulnerability in EchoGuard:

1. **Do not disclose publicly** before contacting the maintainer.
2. Email a description to: vrutulpatel25@gmail.com with subject "EchoGuard Security Disclosure".
3. Allow 30 days for a response and coordinated disclosure.
4. Credit will be given in the release notes unless you prefer anonymity.

---

## License and Liability

EchoGuard is provided under the MIT license "as is" without warranty. The authors accept no liability for decisions made based on EchoGuard's output. Users assume full responsibility for applying detection results in their specific context.

---

## Acknowledgment

By using, modifying, or distributing EchoGuard, you agree to use it only for lawful purposes consistent with the intended use cases described above. This is not a legally binding contract, but a statement of community expectations.
