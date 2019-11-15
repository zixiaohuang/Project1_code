import matplotlib.pyplot as plt

if __name__ == '__main__':
    color = ['aqua', 'r','g', 'pink', 'y', 'darkorange']
    plt.figure()
    plt.xlabel("Training Degrees")
    plt.ylabel("Score")
    plt.xlim((1,10))
    plt.ylim((0.75,1.0))

    #l1 precision
    x=[1,2,3,4,5,6,7,8,9]
    y_preci=[0.85,0.86,0.87,0.90,0.91,0.91,0.77,0.77,0.78]
    plt.plot(x,y_preci,'o-',color='red',label='Precision Penalty=L1')
    #l2 precision
    y_preci2 = [0.83, 0.84, 0.87, 0.89, 0.89, 0.87, 0.82, 0.84, 0.84]
    plt.plot(x, y_preci2, '--', color='coral', label='Precision Penalty=L2')

    #l1_recall
    y_recall = [0.84, 0.85, 0.87, 0.89, 0.90, 0.91, 0.77, 0.77, 0.77]
    plt.plot(x, y_recall, 'o-', color='goldenrod', label='Recall Penalty=L1')
    # l2_recall
    y_preci = [0.83, 0.84, 0.87, 0.89, 0.89, 0.87, 0.82, 0.81, 0.81]
    plt.plot(x, y_preci, '-.', color='wheat', label='Recall Penalty=L2')

    # l1 F1-score
    y_recall = [0.84, 0.85, 0.87, 0.89, 0.90, 0.91, 0.77, 0.77, 0.77]
    plt.plot(x, y_recall, 'o-', color='green', label='F1-score Penalty=L1')
    # l2_recall
    y_preci = [0.83, 0.84, 0.87, 0.89, 0.89, 0.87, 0.82, 0.80, 0.80]
    plt.plot(x, y_preci, ':', color='lightgreen', label='F1-score Penalty=L2')
    plt.legend()
    plt.show()
