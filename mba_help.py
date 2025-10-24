"""
Market Basket Analysis Metrics Explained
=======================================

Support
-------
- **Definition**: Frequency of itemset appearing in all transactions
- **Formula**: Support(X) = (Transactions containing X) / (Total transactions)
- **Interpretation**: 
  - High support = Common combination
  - Low support = Rare but potentially interesting

Confidence
----------
- **Definition**: Likelihood of consequent given antecedent
- **Formula**: Confidence(X→Y) = Support(X ∪ Y) / Support(X)
- **Interpretation**: 
  - High confidence = Strong predictability
  - Watch for misleading high confidence with common consequents

Lift
----
- **Definition**: Strength of association compared to random chance
- **Formula**: Lift(X→Y) = Support(X ∪ Y) / (Support(X) * Support(Y))
- **Interpretation**:
  - Lift > 1 = Positive association
  - Lift = 1 = Independent items
  - Lift < 1 = Negative association

Conviction
----------
- **Definition**: Measure of implication direction
- **Formula**: Conviction(X→Y) = (1 - Support(Y)) / (1 - Confidence(X→Y))
- **Interpretation**:
  - ∞ = Perfect association
  - 1 = Independence

Key Insights
------------
1. **Support** helps identify common patterns
2. **Confidence** shows prediction strength
3. **Lift** reveals true interestingness
4. Always consider business context with metrics
"""
