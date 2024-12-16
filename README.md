## Installation

Follow these steps to install the project:

1. Clone the repository:

   ```bash
   git clone https://github.com/alekseevpavel04/s_hw2.git
   ```

2. Navigate to the project directory:

   ```bash
   cd s_hw2
   ```

3. Create a virtual environment:

   ```bash
   python3 -m venv venv
   ```

4. Install PyTorch (to avoid compatibility issues):

   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

5. Install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Download the necessary data (models, language models, and vocabulary):

   ```bash
   sh ./download_all_we_need.sh
   ```

## Quick Inference with Metrics Calculation

Run the following command to perform inference and calculate metrics:

```bash
python inference.py
