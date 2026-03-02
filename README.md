# CNN Architectures Benchmarking

This project implements and compares multiple state-of-the-art pre-trained CNN architectures including ResNet, DenseNet, MobileNet, and EfficientNet using transfer learning. The objective is to evaluate performance, computational efficiency, and model size on a custom dataset. The project highlights the architectural innovations of each model and benchmarks their performance under identical training settings.

## Useful GitHub Commands for Contributors

Here are some essential Git and GitHub commands to help you contribute to this project:

### 1. Forking and Cloning
- **Clone your fork**:
  ```bash
  git clone https://github.com/vasu-rguktn/missionX.git
  cd missionX
  ```
- **Add the original repository as an upstream remote**:
  ```bash
  git remote add upstream https://github.com/vasu-rguktn/missionX.git
  ```

### 2. Branching
- **Create a new branch** for your feature or bug fix:
  ```bash
  git checkout -b feature/your-feature-name
  ```
- **Switch to an existing branch**:
  ```bash
  git checkout branch-name
  ```

### 3. Making Changes
- **Check the status** of your files:
  ```bash
  git status
  ```
- **Stage your changes** for commit:
  ```bash
  git add .
  # Or stage specific files: git add file1 file2
  ```
- **Commit your changes**:
  ```bash
  git commit -m "Add a clear and descriptive commit message"
  ```


### 5. Pushing Changes and Pull Requests
- **Push your branch** to your forked repository on GitHub:
  ```bash
  git push origin feature/your-feature-name
  ```
- After pushing, visit the repository on GitHub to open a **Pull Request**.

### 6. Updating Your Branch (Rebasing)
If `main` has moved forward, you might need to rebase your branch:
  ```bash
  git checkout feature/your-feature-name
  git rebase main
  # Resolve any conflicts, then:
  # git add .
  # git rebase --continue
  ```