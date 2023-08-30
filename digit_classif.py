"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


from util_fun import load_and_visualize_digits, preprocess_data, train_and_predict, visualize_predictions, evaluate_classifier

def main():
    digits = load_and_visualize_digits()
    data = preprocess_data(digits)
    X_test, predicted, y_test = train_and_predict(data, digits.target)
    visualize_predictions(X_test, predicted)
    evaluate_classifier(y_test, predicted)

if __name__ == "__main__":
    main()

