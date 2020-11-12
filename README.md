# AI-enhancer
* ## A deep learning algorithm to predict enhancer regions from DNA sequence <h2> 
* ## Workflow <h3> 
![GitHub Logo](/images/Model_plot.png)

# Install packages
  Users need to install python (https://www.python.org/downloads/) and some python packages:
   * [tensorflow]
   * [keras]
   * [numpy]
   * [pandas]
   * [Bio]
   * [scikit-learn]
   
# Data preparation
 1. Add sequence fasta files into folder Data <h4> 
 2. Run python script “sequenceEncode.py” to train and validate the AI-enhancer model. The results will be stored in folder EncodeData. <h4>
 3. Run python script "model_building.py". The well-trained AI-enhancer model will be stored in the folder "ModelOutput" for follow-up enhancer prediction. <h4>
 4. AI-enhancer will provide enhancer prediction with probability scores to rank all candidate enhancers, which will facilitate the follow-up experimental validation (e.g. CRISPR/Cas9 knockout of predicted enhancers). <h4>
