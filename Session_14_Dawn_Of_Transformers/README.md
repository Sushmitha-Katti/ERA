# English to Italian Translation with Encoder-Decoder 📚

[![Notebook](https://img.shields.io/badge/Notebook-Open-blue.svg)](https://www.kaggle.com/code/sushmithakatti/s14-transformers)
[![Code](https://img.shields.io/badge/Code-View%20on%20GitHub-green.svg)](https://github.com/Sushmitha-Katti/ERA/tree/main/Session_14_Dawn_Of_Transformers/modular)

## Assignment Overview
The goal of this assignment was to train an English to French translation model using the OPUS book translation dataset. The primary objective was to achieve a final loss of less than 4 during training in 10 epochs.

> Note: Not to use any optimization techniques

## Data Source and Preprocessing 📝
- **Dataset:** The dataset used for this task was the `OPUS book` translation dataset from hugging face.

## Model Architecture 🧠
- **Encoder-Decoder:** This model utilizes the encoder-decoder architecture for sequence-to-sequence translation tasks.
- **Model Size:** The model consists of approximately 70.2 million parameters.

## Training Configuration ⚙️
- **Batch Size:** Training was conducted with a batch size of `16.`
- **Epochs:** The training process spanned `10 epochs.`
- Used CrossEntrophy loss

### Learning Rate Scheduler
- No scheduler was used
- **Optimizer:** Used the ADAM optimizer.

## Training Results 📊
- **Final Loss:** The training loss reached `2.12`.
- Average time taken for 1 epoch is 14-15 min. 

### Training Plots

<table>
  <tr>
    <td align="center"><b>Average Train Loss Per Batch</b></td>
   
  <tr>
    <td><img src="assets/train_loss.png" alt="Average Train Loss Per Batch" width="300"/></td>
  </tr>
</table>

### Evaluation Metrics

<table>
  <tr>
    <td align="center"><b>BLEU Score</b></td>
    <td align="center"><b>CER Validation</b></td>
    <td align="center"><b>WER Validation</b></td>
    
  </tr>
  <tr>
    <td><img src="assets/validation_BLEU.png" alt="BLEU Score" width="300"/></td>
    <td><img src="assets/validation_cer.png" alt="CER Validation" width="300"/></td>
    <td><img src="assets/validation_wer.png" alt="WER Validation" width="300"/></td>
  </tr>
</table>







