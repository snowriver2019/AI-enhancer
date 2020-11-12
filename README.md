# AI-enhancer
* ## A deep learning algorithm to predict enhancer regions from DNA sequence <h2> 
* ## Workflow <h3> 
![GitHub Logo](/images/Model_plot.png)
* ## Manual <h3> 
  * ### Step 1: Users need to install python and some python packages:
   tensorflow<h4>  
   keras<h4> 
   numpy<h4> 
   pandas<h4> 
   Bio<h4> 
   
  * ### Step 2: Users need to add reference genomes (hg19 or mm10) and all necessary genomic data into Data folder. <h4> 
  * ### Step 3: Users need to run python script “sequenceEncode.py” to train and validate the AI-enhancer model. <h4>
  * ### Step 4: Users need to run python script "model_building.py". The well-trained AI-enhancer model will be stored in the folder "ModelOutput" for follow-up enhancer prediction. <h4>
  * ### Step 5: AI-enhancer will provide enhancer prediction with probability scores to rank all candidate enhancers, which will facilitate the follow-up experimental validation (e.g. CRISPR/Cas9 knockout of predicted enhancers). <h4>
