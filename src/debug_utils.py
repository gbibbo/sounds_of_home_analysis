# src/debug_utils.py

class DebugStats:
    def __init__(self):
        self.max_probability = 0
        self.predictions_above_threshold = 0
        self.total_predictions = 0
        self.class_probabilities = {}
        
    def update(self, class_label, probability, threshold):
        self.total_predictions += 1
        self.max_probability = max(self.max_probability, probability)
        if probability >= threshold:
            self.predictions_above_threshold += 1
        
        # Track maximum probability per class
        current_max = self.class_probabilities.get(class_label, 0)
        self.class_probabilities[class_label] = max(current_max, probability)

    def print_summary(self, threshold):
        print("\nDEBUG SUMMARY:")
        print(f"Total predictions processed: {self.total_predictions}")
        print(f"Predictions above threshold ({threshold}): {self.predictions_above_threshold}")
        print(f"Maximum probability found: {self.max_probability:.4f}")
        print("\nTop 10 classes by probability:")
        top_classes = sorted(self.class_probabilities.items(), key=lambda x: x[1], reverse=True)[:10]
        for class_label, prob in top_classes:
            print(f"  {class_label}: {prob:.4f}")