# Contributing to faster-whisper-rs

Thank you for your interest in contributing!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone --recursive https://github.com/manmal/faster-whisper-rs
   cd faster-whisper-rs
   ```

2. **Install prerequisites**
   - Node.js v18+
   - Rust (via [rustup](https://rustup.rs/))
   - Platform-specific: See README for details

3. **Download prebuilt CTranslate2**
   ```bash
   ./scripts/download-prebuilt.sh
   ```

4. **Build**
   ```bash
   npm install
   npm run build
   ```

5. **Download a test model**
   ```bash
   mkdir -p models && cd models
   git lfs install
   git clone --depth 1 https://huggingface.co/Systran/faster-whisper-tiny tiny
   ```

6. **Run tests**
   ```bash
   npm test           # Smoke test
   npm run test:unit  # Unit tests
   npm run test:e2e   # E2E tests
   npm run test:all   # All tests
   ```

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Make your changes
3. Add tests if applicable
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Reporting Issues

Please include:
- OS and architecture
- Node.js version
- Rust version
- Steps to reproduce
- Error messages (full stack trace)

## Code Style

- Rust: Follow standard Rust conventions (`cargo fmt`)
- JavaScript: Standard style, no semicolons optional
- Commits: Conventional commits preferred

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
