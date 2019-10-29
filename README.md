# AI-enhancer
* ## A deep learning algorithm to predict enhancer regions <h2> 
* ## Workflow <h3> 
![GitHub Logo](/images/Enhancer_CNNmodel.png)
* ## Manual <h3> 
  * ### Step 1: Users need to add reference genomes (hg19 or mm10) and all necessary genomic data into the designated file path for deep learning model training and validation. <h4> 
  * ### Step 2: Users need to run python script “main.py” to train and validate the AI-enhancer model. <h4>
  * ### Step 3: After training and validation steps, the well-trained AI-enhancer model will be stored in the folder "Model" for follow-up enhancer prediction. <h3>
  * ### Step 4: AI-enhancer will provide enhancer prediction with probability scores to rank all candidate enhancers, which will facilitate the follow-up experimental validation (e.g. CRISPR/Cas9 knockout of predicted enhancers). <h4>
