Project 1 sample data pack

Files:
1. training_data.csv
   - 1500 labeled rows
   - Use this to train and test a churn prediction model
   - Target column: churn
     * 1 = churned
     * 0 = retained

2. input_data.csv
   - 300 unlabeled rows
   - Use this as new incoming batch data for inference
   - This file does not contain the churn column because your model should predict it

Columns:
- customer_id: unique customer identifier
- age: customer age
- monthly_spend: current recurring spend
- tenure_months: months as a customer
- support_tickets: number of support tickets raised
- last_login_days: days since last login
- contract_type: Monthly, Quarterly, or Annual
- payment_method: Card, Bank Transfer, UPI, or Wallet
- region: customer region
- num_products: number of subscribed products
- discount_used: 1 if discount applied, else 0
- avg_session_minutes: average session length

Suggested local folders:
- data/training_data.csv
- input/input_data.csv
