The content in this repository is the code and data used to benchmark two different algorithms (Evolution Stratefy and Differential Evolution) in the same environment from the Gymnasium python library. 

In the code folder there is the source code utilized to generate all the data needed to compare the algorithms, included the problem environment, the optimization of the algorithms and of course the algorithms themselves.
In the data folder there are all the csv files generated using the source code, divided into the types of environments: discrete or continuous. There is also another folder containing the data of the same algorithm compared in different envoronments, to showcase the no free lunch theorem.
In the plot folder there are all the plots generated using the data. The fitness plots show the fitness of all the individual runs of an algorithm in the environment, the reward plots show the reward per step in the best run for that algorithm in that environment.
The benchmarks plots show the performances of different algorithms in the same environment and they were generated using the IOHAnalyzer(https://iohanalyzer.liacs.nl/)
