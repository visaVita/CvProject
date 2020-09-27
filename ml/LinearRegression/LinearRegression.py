import numpy as np
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 计算均方差
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))



def step_gradient(b_cur, w_cur, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2/N) * ((w_cur * x + b_cur) - y)
        w_gradient += (2/N) * x * ((w_cur * x + b_cur) - y)
    new_b = b_cur - (learningRate * b_gradient)
    new_w = w_cur - (learningRate * w_gradient)
    return [new_b, new_w]

# Set w = w' and loop
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    # update for several times
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


def run():
    points = np.genfromtxt("ml\\data.csv", delimiter=",")
    #print(points)
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 500
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}".format(initial_b, initial_w, compute_error_for_line_given_points(initial_b, initial_w, points)))
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".format(num_iterations, b, w, compute_error_for_line_given_points(b, w, points)))

if __name__ == '__main__':
    run()