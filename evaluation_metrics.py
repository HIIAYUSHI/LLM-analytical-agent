import numpy as np
import re
from typing import List

class EvaluationMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_claims = 0
        self.unsupported_claims = 0
        self.total_numerical_refs = 0
        self.validated_numerical_refs = 0
        self.total_insights = 0
        self.consistent_insights = 0
        self.confidence_scores = []
        self.accuracy_scores = []

    def calculate_hallucination_rate(self, insights: str, stats: dict) -> dict:
        """
        Upgraded Hallucination Tracker (NLP-Aware):
        Fixes comma-separation bugs and ignores list indices.
        """
        # PRE-PROCESSING FIX: Remove commas that are sandwiched between digits 
        # (e.g., "1,450.00" becomes "1450.00")
        clean_insights = re.sub(r'(?<=\d),(?=\d)', '', insights)
        
        # Extract ALL numbers from the cleaned text
        numbers_in_text = re.findall(r'-?\d+\.?\d*', clean_insights)
        
        if not numbers_in_text:
            return {"hr": 0.0, "unsupported": 0, "total": 0}

        unsupported_count = 0
        valid_claims_count = 0 
        
        # Flatten all "Truth" values from the stats dictionary
        truth_values = []
        for key, value in stats.get("numeric_summary", {}).items():
            if isinstance(value, dict):
                for sub_k, sub_v in value.items():
                    try:
                        truth_values.append(float(sub_v))
                    except (ValueError, TypeError):
                        pass

        for num_str in numbers_in_text:
            try:
                val = float(num_str)
                
                # LIST INDEX FIX: Ignore exact integers from 1 to 100. 
                # These are almost always bullet points (1., 2., 3.) or rank numbers.
                # If they aren't explicitly in the truth dictionary, we just skip them.
                if val.is_integer() and 1 <= val <= 100 and val not in truth_values:
                    continue 
                    
                valid_claims_count += 1
                
                # FIX 1: Epsilon Tolerance (Checks if within 0.05)
                is_supported = any(abs(val - tv) <= 0.05 for tv in truth_values)
                
                # FIX 3: Derived Percentages (Checks if val/100 matches)
                is_percent_supported = any(abs((val / 100) - tv) <= 0.01 for tv in truth_values)
                
                if not (is_supported or is_percent_supported):
                    unsupported_count += 1
            except ValueError:
                pass 
        
        self.total_claims += valid_claims_count
        self.unsupported_claims += unsupported_count
        
        # Calculate granular HR using only the valid claims
        hr = unsupported_count / valid_claims_count if valid_claims_count > 0 else 0.0
        return {"hr": hr, "unsupported": unsupported_count, "total": valid_claims_count}

    def calculate_ngs(self, insights: str, stats: dict) -> dict:
        hr_data = self.calculate_hallucination_rate(insights, stats)
        ngs = 1.0 - hr_data["hr"]
        return {"ngs": ngs}

    def calculate_acs(self, insights: str, stats: dict) -> dict:
        return {"acs": 1.0} # Placeholder for semantic consistency

    def calculate_cce(self, assigned_confidence: float, actual_accuracy: float) -> dict:
        cce = abs(assigned_confidence - actual_accuracy)
        self.confidence_scores.append(assigned_confidence)
        self.accuracy_scores.append(actual_accuracy)
        return {"cce": cce}

    def calculate_stability(self, outputs: List[str]) -> dict:
        if len(outputs) < 2: return {"stability": 1.0}
        lengths = [len(out) for out in outputs]
        variance = np.var(lengths)
        stability = 1.0 / (1.0 + variance)
        return {"stability": float(stability)}

metrics_tracker = EvaluationMetrics()