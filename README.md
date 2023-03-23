
# Recommending Classes from (USPTO ID manual) based on Text Input.

The objective was to group query text from user to the correct class among the 46 classes mentioned in USPTO ID manual.

## Training the model

<strong>For Training part of this model refer <i> Train.ipynb</i></strong>

I have used Transformers library and finetuned the BERT model on the Data set provided. The model was trained on 90% of the total data and rest 10% was reserved for testing.

After 4 epochs we reached the accuracy of about 90% on our training data.

## Evaluation

I have saved and loaded the trained model and performed Evaluation using the testing data we reserved.
I also Calulated Class level accuracy that is how many samples of each class are actually correct.

We achieved Testing accuracy of about 87%.

## API
<strong>Please refer <i>main.py</i> </strong>

The API is developed in FastAPI using python. User sends a POST request containing the Text or Query and the system recommends the class to which the text belongs with highest Probability. 

As an Additional feature system also returns the second most probable class. Please have a look at example images priovided.  

## Next Steps

In order to improve our model accuracy we can experiment with multiple approaches like Hyperparameter Tuning (GridCVsearch) or using different architecture/models like DistilBERT, RoBERTa, LayoutLM etc   

## Screenshot

![image1](https://user-images.githubusercontent.com/51448962/227155502-c1a1ea5b-f4c2-4988-8c1b-1816edf60de1.jpg)






