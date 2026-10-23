-- Customer Churn Predictor: example analysis queries for MySQL 8+
-- First create a table named customer_churn and import data/customer_churn.csv
-- using MySQL Workbench's Table Data Import Wizard.
-- If your actual column names differ, update these queries.

-- 1. Total customers and observed churn rate
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2)
        AS churn_rate_percent
FROM customer_churn;

-- 2. Churn rate by contract type
SELECT Contract,
       COUNT(*) AS customers,
       SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
       ROUND(100.0 * AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 2)
           AS churn_rate_percent
FROM customer_churn
GROUP BY Contract
ORDER BY churn_rate_percent DESC;

-- 3. Churn rate by payment method
SELECT PaymentMethod,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 2)
           AS churn_rate_percent
FROM customer_churn
GROUP BY PaymentMethod
ORDER BY churn_rate_percent DESC;

-- 4. Churn rate by tenure group
SELECT
    CASE
        WHEN tenure < 12 THEN '0-11 months'
        WHEN tenure < 24 THEN '12-23 months'
        WHEN tenure < 48 THEN '24-47 months'
        ELSE '48+ months'
    END AS tenure_group,
    COUNT(*) AS customers,
    ROUND(100.0 * AVG(CASE WHEN Churn = 'Yes' THEN 1.0 ELSE 0.0 END), 2)
        AS churn_rate_percent
FROM customer_churn
GROUP BY tenure_group
ORDER BY churn_rate_percent DESC;

-- 5. Average monthly charges by churn status
SELECT Churn, COUNT(*) AS customers,
       ROUND(AVG(MonthlyCharges), 2) AS average_monthly_charges
FROM customer_churn
GROUP BY Churn;

-- 6. Customers with high monthly charges and month-to-month contracts
SELECT customerID, tenure, Contract, PaymentMethod, MonthlyCharges, Churn
FROM customer_churn
WHERE Contract = 'Month-to-month'
  AND MonthlyCharges >= (SELECT AVG(MonthlyCharges) FROM customer_churn)
ORDER BY MonthlyCharges DESC
LIMIT 50;
