import numpy as np


class ClassificationReport:
    def __init__(
        self,
        class_name: [list, np.ndarray] = None,
        digits: int = 3,
    ) -> None:
        self.class_name = class_name
        self.digits = digits
        self.epsilon = 1e-9
        self.reset()

    def compute(self, y_true: [list, np.ndarray], y_pred: [list, np.ndarray]) -> None:
        if not isinstance(y_true, (list, np.ndarray)):
            raise TypeError("'y_true' type is 'list' or 'np.ndarray'")
        if not isinstance(y_pred, (list, np.ndarray)):
            raise TypeError("'y_pred' type is 'list' or 'np.ndarray'")

        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)

        for t, p in zip(y_true, y_pred):
            self.confusion_matrix[t, p] += 1
            self.sample_per_class[t] += 1

        if self.binary:
            self.tp[0] = self.confusion_matrix[0, 0]
            self.fn[0] = self.confusion_matrix[0, 1]
            self.fp[0] = self.confusion_matrix[1, 0]
            self.tn[0] = self.confusion_matrix[1, 1]
        else:
            for i in range(len(self.class_name)):
                self.tp[i] = self.confusion_matrix[i, i]
                self.fn[i] = self.confusion_matrix[i, :].sum() - self.tp[i]
                self.fp[i] = self.confusion_matrix[:, i].sum() - self.tp[i]
                self.tn[i] = (
                    self.confusion_matrix.sum() - self.fn[i] - self.fp[i] - self.tp[i]
                )

    def report_dict(self) -> dict:
        self._accuracy()
        self._precision()
        self._recall()
        self._f1_score()

        report = dict()

        if self.binary:
            report[self.class_name[0]] = {
                "precision": self.precision[0],
                "recall": self.recall[0],
                "f1_score": self.f1_score[0],
                "n_sample": self.sample_per_class[0],
            }
            report["accuracy"] = self.accuracy[0]
            report["total_sample"] = self.sample_per_class.sum()

            return report
        else:
            for i in range(len(self.class_name)):
                report[self.class_name[i]] = {
                    "precision": self.precision[i],
                    "recall": self.recall[i],
                    "f1_score": self.f1_score[i],
                    "n_sample": self.sample_per_class[i],
                }
        report["accuracy"] = self.accuracy[0]
        report["total_sample"] = self.sample_per_class.sum()
        return report

    def print_report(self) -> str:
        self._accuracy()
        self._precision()
        self._recall()
        self._f1_score()

        header = ["precision", "recall", "f1_score", "n_sample"]
        max_class_name_length = max(
            [len(name) for name in self.class_name] + [len("accuracy")]
        )
        header_fmt = "{:>{length}s}" + "{:>10s}" * len(header)
        report = header_fmt.format(
            "",
            *header,
            length=max_class_name_length,
        )
        report += "\n"

        row_fmt = "{:>{length}s}" + "{:>10.{digits}f}" * (len(header) - 1) + "{:>10d}"

        if self.binary:
            report += row_fmt.format(
                *[
                    self.class_name[0],
                    self.precision[0],
                    self.recall[0],
                    self.f1_score[0],
                    self.sample_per_class[0],
                ],
                length=max_class_name_length,
                digits=self.digits,
            )
            report += "-" * (max_class_name_length + 10 * len(header)) + "\n"
            report += row_fmt.format(
                *[
                    "accuracy",
                    0,
                    0,
                    self.accuracy[0],
                    self.sample_per_class.sum(),
                ],
                length=max_class_name_length,
                digits=self.digits,
            )
            return report
        else:
            for i in range(len(self.class_name)):
                report += row_fmt.format(
                    *[
                        self.class_name[i],
                        self.precision[i],
                        self.recall[i],
                        self.f1_score[i],
                        self.sample_per_class[i],
                    ],
                    length=max_class_name_length,
                    digits=self.digits,
                )
                report += "\n"

        report += "-" * (max_class_name_length + 10 * len(header)) + "\n"
        report += row_fmt.format(
            *[
                "accuracy",
                0,
                0,
                self.accuracy[0],
                self.sample_per_class.sum(),
            ],
            length=max_class_name_length,
            digits=self.digits,
        )

        return report

    def reset(self) -> None:
        self.tp = np.zeros(len(self.class_name))
        self.tn = np.zeros(len(self.class_name))
        self.fp = np.zeros(len(self.class_name))
        self.fn = np.zeros(len(self.class_name))
        self.accuracy = np.zeros(1)
        self.precision = np.zeros(len(self.class_name))
        self.recall = np.zeros(len(self.class_name))
        self.f1_score = np.zeros(len(self.class_name))
        self.sample_per_class = np.zeros(len(self.class_name), dtype=np.int64)

        if len(self.class_name) == 1:
            self.confusion_matrix = np.zeros(shape=(2, 2))
            self.binary = True
        else:
            self.confusion_matrix = np.zeros(
                shape=(len(self.class_name), len(self.class_name))
            )
            self.binary = False

    def _accuracy(self) -> None:
        if self.binary:
            self.accuracy[0] = (self.tp[0] + self.tn[0]) / (
                self.tp[0] + self.tn[0] + self.fp[0] + self.fn[0] + self.epsilon
            )
        else:
            self.accuracy[0] = self.tp.sum() / self.sample_per_class.sum()

    def _precision(self) -> None:
        if self.binary:
            self.precision[0] = self.tp[0] / (self.tp[0] + self.fp[0] + self.epsilon)
        else:
            for i in range(len(self.class_name)):
                self.precision[i] = self.tp[i] / (
                    self.tp[i] + self.fp[i] + self.epsilon
                )

    def _recall(self) -> None:
        if self.binary:
            self.recall[0] = self.tp[0] / (self.tp[0] + self.fn[0] + self.epsilon)
        else:
            for i in range(len(self.class_name)):
                self.recall[i] = self.tp[i] / (self.tp[i] + self.fn[i] + self.epsilon)

    def _f1_score(self) -> None:
        if self.binary:
            self.f1_score[0] = (2 * self.precision[0] * self.recall[0]) / (
                self.precision[0] + self.recall[0] + self.epsilon
            )
        else:
            for i in range(len(self.class_name)):
                self.f1_score[i] = (2 * self.precision[i] * self.recall[i]) / (
                    self.precision[i] + self.recall[i] + self.epsilon
                )
