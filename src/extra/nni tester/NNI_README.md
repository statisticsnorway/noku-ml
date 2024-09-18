# NNI Testing

Norsk versjon:

- [Norsk](NNI_README_NO)

This program aims to evaulate the performance of the NNI model we use in the production of NØKU statistics. This is done by the use of a bootstrap method that treats several years of reg_type 01 foretak schema responses as the population. A random sample of this dataset is randomly choose over X number of iterations in order to behave as a sample and 'give' information to the remaining rows in the population. Results are evaluated by comparing predicted sales to actual sales revenue reported my the survey responders. 

Here is a run through of the analysis:

## Histogram Analysis:

Not strictly necessary for the NNI testing, however provides important context for the skewness and distribution of data on a '3 - siffer' industry level. This has important implications for how many foretak we should be mandually editing/checking, but also for how many skjema we should be delivering in any case. 


![image](https://github.com/user-attachments/assets/ce08705b-96da-4cc1-b758-671e1fe71e8b)


## Bootstrap testing

Bootstrap testing was employed to access the performance of the model, with the aim to get some insight into how it may perform year after year, rather than just a single iteration. 

![image](https://github.com/user-attachments/assets/839eaefa-2a58-4c61-831a-c1c9d0931c5c)

#### Results:

The MAE of the results is actually quite reasonable, although on some iterations it was signficantly higher. A closer look at the residual errors raises some concern however. There appears to be some large outliers and in addition to this there seems to be some bias in the residuals. 

## Evaluation of a new model. 

A new model was evaluated over two years (only two years of published data available currently in the cloud platform). This model didnt change a lot of what we currently use, as the main goal of the analysis was to assess the feasibility of using reg_type 2 companies so that it may become possible to no longer deliver schema to reg_type 1 companies. However we used a knn = X method, rather than a single neighbor. The same features were used to train the model, so there is potential to add more features in the future. 

#### Results:

![image](https://github.com/user-attachments/assets/63041104-08f2-40bd-b338-158235e958e1)

The MAE was in fact larger in this case, however the residual plot seems to be vasty improved. 

An A/B test was also produced (that cannot be shared here due to confidentiality rules) which compared our actual published results to what they would have been using the new method. The actual change was insignficant with most industries - however some industries had large differences. So some extra controls would need to be added. 

## Areas for improvment

- Add controls for outliers in the results.
- build a more robust model for replacing the NNI model
