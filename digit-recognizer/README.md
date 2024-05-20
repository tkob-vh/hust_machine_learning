# 文件结构
```bash
digit-recognizer
├── data # The training data, test data and the data to be submitted.
│   ├── sample_submission.csv
│   ├── submission_dt.csv
│   ├── submission_knn.csv
│   ├── submission_mlp.csv
│   ├── test.csv
│   └── train.csv
├── job.slurm # The script to run the model
├── log # The graphs and standard output.
│   ├── decisiontree
│   │   ├── log.txt
│   │   ├── precision_by_class.png
│   │   └── recall_by_class.png
│   ├── knn
│   │   ├── k_accuracy_plot.png
│   │   └── log.txt
│   └── mlp
│       ├── accuracy.png
│       ├── balanced accuracy.png
│       ├── log.txt
│       ├── loss.png
│       ├── precision.png
│       └── recall.png
├── main.py # The main program
├── model # The model program
│   ├── decisontree.py
│   ├── knn.py
│   └── mlp.py
├── preprocess.py # The data proprocessing program
├── README.md
└── xxx_机器学习结课报告.pdf # The report

6 directories, 25 files
```