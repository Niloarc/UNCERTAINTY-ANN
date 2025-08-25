Data collection and distributions
Input variables for the study are based on previous literature which highlighted financial, technical, and supply chain factors are the most important on fuel MSP. Historical data for financial, technical, and supply chain variables will be sourced from literature, fit to probability distributions (e.g., normal, gamma, beta) based on the highest p-value in Kolmogorov-Smirnov tests. When data is limited, a uniform distribution will be employed to avoid value concentration around specific peaks. Feature are presented in Table1, all these inputs are based on literature of this study [1]. There are total of 17 features, which under financial variables there are, Equity (%), Loan interest, Loan term (years), Discount rate (%), Income tax rate (%). Nine technical variables were selected including, EH enzyme loading (mg/g), PT acid loading (g/g), EH cellulose to glucose (%), EH % solids (%), FERM contamination losses (%), FERM xylose to ethanol (%), PT xylan to xylose (%), FERM arabinose to ethanol (%), PT glucan to glucose (%). Finally for supply chain variables include Feedstock cost ($/dry ton), Ammonia cost ($/ton), Glucose cost ($/lb).
Table 1.
Features and input variables for fuel pathway. 
Variable	Category	Distribution	Boundary	Base
Equity	Financial	Uniform	[30, 60]	40
Discount Rate	Financial	Beta	[0.01, 0.1]	0.08
Income Tax Rate	Financial	Uniform	[5, 15]	10
Loan Interest	Financial	Uniform	[10, 30]	10
Loan Term	Financial	Uniform	[0.011, 0.038]	25
EH % Solids	Technical	Uniform	[75, 95]	20
FERM Arabinose to Ethanol	Technical	Uniform	[0, 6]	0.024
FERM Contamination Losses	Technical	Uniform	[75, 90]	90
FERM Xylose to Ethanol	Technical	Uniform	[80, 92]	20
PT Acid Loading	Technical	Uniform	[0, 85]	3
PT Glucan to Glucose	Technical	Uniform	[6, 12	85
EH Cellulose to Glucose	Technical	Uniform	[75, 95]	90
EH Enzyme Loading	Technical	Uniform	[10, 30]	85
PT Xylan to Xylose	Technical	Uniform	[80,92]	9.9
Glucose Cost	Supply chain	Uniform	[0.306, 0.463]	0.3826
Ammonia Cost	Supply chain	Gamma	[127, 575]	317
Feedstock Cost	Supply chain	Uniform	[84, 100]	84.45

ML models and randomized data
Random data was generated for variables following the defined distribution and boundary limits. The variables then were fed to ASPEN Plus model to generate MSP using a through TEA and chemical energy balance analysis. The random variables were used as features and the calculated MSP as output to train the ML models. Training and test set were divided by 70-30 proportion. Using outputs from initial simulations, 2 ML models will be compared, including shallow neural network (NN) and support vector machine (SVM) to perform uncertainty analyses. The ML framework will support sample sizes up to 50,000, drastically enhancing the capacity for comprehensive TEA. The distribution of data for training validation and test set is 60-25-15, which is code we only divide to training and test set but in neural network, when we are splitting the data a portion of it is split for validation before testing.
history = best_model.fit(X_train, y_train, 
                         epochs=best_params['epochs'], 
                         batch_size=best_params['batch_size'], 
                         validation_split=0.2)

 
Fig.1. Data distribution for training, validation, and testing.
A t-SNE (t-Distributed Stochastic Neighbor Embedding) plot was used to visualize features, which is a non-linear dimensionality reduction technique primarily used for visualizing high-dimensional data by preserving local relationships and revealing clusters. Unlike PCA, which focuses on maximizing variance through linear transformations, t-SNE emphasizes capturing the similarity between nearby points, making it ideal for understanding the local structure of data. While PCA is efficient and interpretable for feature reduction or preprocessing, t-SNE is better suited for finding patterns between the input variables and MFSP.
