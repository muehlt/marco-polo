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


        self.main_color = '#1432F4'       # blue
        self.secondary_color = '#F41432'  # red
        self.tertiary_color = '#32F414'   # green

    def scatter_plot_recall(self, threshold):
        plt.figure(1)
        ax = plt.axes()
        ax.scatter(self.keys, self.c_recall.values(), label="Corpus", color=self.main_color)
        ax.scatter(self.keys, self.s_recall.values(), label="Summaries", color=self.tertiary_color)
        ax.set_title(f"Recall \n t = {threshold}")
        ax.set_ylabel('Recall score')
        ax.set_xlabel('Query IDs')
        ax.legend()
        plt.show()

    def scatter_plot_precision(self, threshold):
        plt.figure(2)
        ax = plt.axes()
        ax.scatter(self.keys, self.c_precision.values(), label="Corpus", color=self.main_color)
        ax.scatter(self.keys, self.s_precision.values(), label="Summaries", color=self.tertiary_color)
        ax.set_title(f"Precision \n t = {threshold}")
        ax.set_ylabel('Precision score')
        ax.set_xlabel('Query IDs')
        ax.legend()
        plt.show()

    def scatter_plot_fscore(self, threshold):
        plt.figure(3)
        ax = plt.axes()
        ax.scatter(self.keys, self.c_fscore.values(), label="Corpus", color=self.main_color)
        ax.scatter(self.keys, self.s_fscore.values(), label="Summaries", color=self.secondary_color)
        ax.set_title(f"F-Score \n t = {threshold}")
        ax.set_ylabel('F-Score')
        ax.set_xlabel('Query IDs')
        ax.legend()
        plt.show()

    def calculateDiff(self, corpus, summarized):
        diff_list = []
        for key in corpus.keys():
            diff_list.append(corpus[key] - summarized[key])
        return diff_list

    def boxplot_recall_differences(self):
        plt.figure(4)
        diff_recall = self.calculateDiff(self.c_recall, self.s_recall)
        maximum = max(diff_recall)
        minimum = min(diff_recall)
        max_diff = max(abs(maximum), abs(minimum))
        plt.boxplot(diff_recall, vert=False)
        plt.xlim(-max_diff-0.05,max_diff+0.05)
        plt.title("Recall differences")
        plt.show()

    def boxplot_precision_differences(self):
        plt.figure(5)
        diff_precision = self.calculateDiff(self.c_precision, self.s_precision)
        maximum = max(diff_precision)
        minimum = min(diff_precision)
        max_diff = max(abs(maximum), abs(minimum))
        plt.boxplot(diff_precision, vert=False)
        plt.xlim(-max_diff - 0.05, max_diff + 0.05)
        plt.title("Precision differences")
        plt.show()

    def boxplot_fscore_differences(self):
        plt.figure(6)
        diff_fscore = self.calculateDiff(self.c_fscore, self.s_fscore)
        maximum = max(diff_fscore)
        minimum = min(diff_fscore)
        max_diff = max(abs(maximum), abs(minimum))
        plt.boxplot(diff_fscore, vert=False)
        plt.xlim(-max_diff - 0.05, max_diff + 0.05)
        plt.title("F-score differences")
        plt.show()