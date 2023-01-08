from matplotlib import pyplot as plt


class Analyzer:

    def __init__(self, c_recall, c_precision, c_fscore,
                 s_recall, s_precision, s_fscore):
        self.c_recall = c_recall
        self.c_precision = c_precision
        self.c_fscore = c_fscore
        self.s_recall = s_recall
        self.s_precision = s_precision
        self.s_fscore = s_fscore
        self.keys = list(range(0, 43))
        self.diff_recall = None

    def scatter_plot_recall(self, threshold):
        plt.figure(1)
        ax = plt.axes()
        ax.scatter(self.keys, self.c_recall.values(), label="Corpus")
        ax.scatter(self.keys, self.s_recall.values(), label="Summaries")
        ax.set_title(f"Recall \n t = {threshold}")
        ax.set_ylabel('Recall score')
        ax.set_xlabel('Query IDs')
        ax.legend()
        plt.show()

    def scatter_plot_precision(self, threshold):
        plt.figure(2)
        ax = plt.axes()
        ax.scatter(self.keys, self.c_precision.values(), label="Corpus")
        ax.scatter(self.keys, self.s_precision.values(), label="Summaries")
        ax.set_title(f"Precision \n t = {threshold}")
        ax.set_ylabel('Precision score')
        ax.set_xlabel('Query IDs')
        ax.legend()
        plt.show()

    def scatter_plot_fscore(self, threshold):
        plt.figure(3)
        ax = plt.axes()
        ax.scatter(self.keys, self.c_fscore.values(), label="Corpus")
        ax.scatter(self.keys, self.s_fscore.values(), label="Summaries")
        ax.set_title(f"F-Score \n t = {threshold}")
        ax.set_ylabel('F-Score')
        ax.set_xlabel('Query IDs')
        ax.legend()
        plt.show()

    def boxplot_recall_differences(self):
        plt.figure(4)
        plt.boxplot(self.diff_recall, vert=False)
        plt.title("Recall differences")
        plt.show()