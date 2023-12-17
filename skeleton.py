#################################
# Your name: matan amichai
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class HypothesisEvaluator(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x = np.random.uniform(0, 1, m)
        x.sort()
        y = np.array([np.random.choice([0, 1], p=self.cal_y_probability_per_x(_)) for _ in x])
        sample = np.stack((x,y),axis=1)
        return sample


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):

        n = int((m_last - m_first)/step)+1
        avg_true_and_emp_error = np.zeros(2*n).reshape(n,2)

        for _ in range(T):
            for m in range(m_first,m_last+1,step):
                sample = self.sample_from_D(m)
                intervals_arry,emp_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
                avg_true_and_emp_error[int((m-m_first)/step)][0] += self.calc_true_error_for_I(intervals_arry)
                avg_true_and_emp_error[int((m-m_first)/step)][1] += emp_error/m
        plt.xlabel("samples")
        plt.ylabel("Empirical Error")
        plt.plot(range(m_first,m_last+1,step), avg_true_and_emp_error[:, 0]/T,marker='o',label="true error")
        plt.plot(range(m_first,m_last+1,step), avg_true_and_emp_error[:, 1]/T,marker='o',label="empirical error")
        plt.legend()
        plt.savefig("1_b.png")

    def cal_emp_true_error_and_emp(self,sample,n,k):
        best_intervals, error_count = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
        x = self.calculate_true_error(best_intervals)
        return error_count / n, x


    def experiment_k_range_erm(self, m, k_first, k_last, step):

        n = int(k_last+1-k_first/step)
        avg_true_and_emp_error = np.zeros(2*n).reshape(n,2)
        sample = self.sample_from_D(m)
        for k in range(k_first,k_last+1,step):
            intervals_arry,emp_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
            avg_true_and_emp_error[int((k-k_first)/step)][0] += self.calc_true_error_for_I(intervals_arry)
            avg_true_and_emp_error[int((k-k_first)/step)][1] += emp_error/m
        plt.title("empirical and true errors as func of k")
        plt.plot(range(k_first,k_last+1,step), avg_true_and_emp_error[:, 0],marker='o',label="true error")
        plt.plot(range(k_first,k_last+1,step), avg_true_and_emp_error[:, 1],marker='o',label="empirical error")
        plt.legend()
        plt.savefig("1_c.png")
        return  np.argmin(avg_true_and_emp_error[:, 1])*step


    def cross_validation(self, m):

        train_set = self.sample_from_D(int(m*0.8))
        test_data = self.sample_from_D(int(m*0.2))
        hypothesis = []
        emp_error_on_test = np.zeros(10,dtype=float)
        for k in range(1,11):
            hypothesis.append(intervals.find_best_interval(train_set[:, 0], train_set[:, 1], k)[0])
            emp_error_on_test[k-1] = (self.cal_emp_error(hypothesis[k-1],test_data))
        min_emp_error_on_test = np.argmin(emp_error_on_test)
        return  hypothesis[min_emp_error_on_test]


    def if_x_in_intervals(self,intervals,x):
        for i in intervals:
            if i[0]<=x<=i[1]:
                return True
        return False


    def zero_one_lost(self,intervals, x, y):
        x_in_intrevals = self.if_x_in_intervals(intervals,x)
        if x_in_intrevals :
            if y == 1:
                return 0
            else:
                return 1
        else:
            if y == 1:
                return 1
            else:
                return 0

    def cal_emp_error(self,hypothesis,data):
        sum = 0
        for _ in data:
            sum += self.zero_one_lost(hypothesis,_[0],_[1])
        return  sum/len(data)


    def calc_true_error_for_I(self,l):
        overlap_high_probability = 0
        overlap_low_probability = 0
        high_probability = np.array([(0, 0.2), (0.4, 0.6), (0.8, 1)])
        low_probability = np.array([(0.2, 0.4), (0.6, 0.8)])
        overlap_high_probability += self.interval_overlap(np.asarray(l),high_probability)
        overlap_low_probability += self.interval_overlap(np.asarray(l),low_probability)
        return overlap_high_probability * 0.2 + overlap_low_probability * 0.9 + (0.6 - overlap_high_probability) * 0.8 + (
                    0.4 - overlap_low_probability) * 0.1



    def cal_overlap_with_interval_array_of_interval(self,intervals1,array_of_interval):
        sum = 0;
        for intervals in array_of_interval:
            sum += self.interval_overlap(intervals1,intervals)

        return sum
    def interval_overlap(self,intervals1, intervals2):
        overlap = 0
        for i in intervals1:
            for j in intervals2:
                overlap += max(0, min(i[1], j[1]) - max(i[0], j[0]))
        return overlap

    def cal_y_probability_per_x(self,x):
        if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
            return [0.2, 0.8]
        return [0.9, 0.1]

    #################################


if __name__ == '__main__':
    ass = HypothesisEvaluator()
    ass.sample_from_D(10)
    #ass.experiment_m_range_erm(10, 100, 5, 3, 100)

    #ass.experiment_k_range_erm(1500, 1, 10, 1)
    print(ass.cross_validation(1500))

