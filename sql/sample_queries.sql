-- ============================================================================
-- Automotive Data Pipeline - Sample Analytical Queries
-- ============================================================================
-- Useful queries for portfolio demonstration and data exploration.
-- ============================================================================

-- 1. Top 10 brands by number of listings
SELECT brand, COUNT(*) AS listing_count, 
       ROUND(AVG(selling_price), 0) AS avg_price
FROM cleaned_car_listings
GROUP BY brand
ORDER BY listing_count DESC
LIMIT 10;

-- 2. Average price by fuel type
SELECT fuel, 
       COUNT(*) AS count,
       ROUND(AVG(selling_price), 0) AS avg_price,
       ROUND(MIN(selling_price), 0) AS min_price,
       ROUND(MAX(selling_price), 0) AS max_price
FROM cleaned_car_listings
GROUP BY fuel
ORDER BY avg_price DESC;

-- 3. Price trend by production year
SELECT year, 
       COUNT(*) AS listings,
       ROUND(AVG(selling_price), 0) AS avg_price,
       ROUND(AVG(km_driven), 0) AS avg_mileage
FROM cleaned_car_listings
GROUP BY year
ORDER BY year DESC;

-- 4. Top 10 most expensive car models (average)
SELECT brand || ' ' || model AS car,
       COUNT(*) AS listings,
       ROUND(AVG(selling_price), 0) AS avg_price
FROM cleaned_car_listings
GROUP BY brand, model
HAVING COUNT(*) >= 5
ORDER BY avg_price DESC
LIMIT 10;

-- 5. Transmission type distribution with price comparison
SELECT transmission,
       COUNT(*) AS count,
       ROUND(AVG(selling_price), 0) AS avg_price,
       ROUND(AVG(km_driven), 0) AS avg_km
FROM cleaned_car_listings
GROUP BY transmission;

-- 6. Owner type impact on price
SELECT owner,
       COUNT(*) AS count,
       ROUND(AVG(selling_price), 0) AS avg_price
FROM cleaned_car_listings
GROUP BY owner
ORDER BY avg_price DESC;

-- 7. Monthly listing trends (from forecasting table)
SELECT year_month, listing_count, 
       ROUND(avg_price, 0) AS avg_price
FROM monthly_forecasting_data
ORDER BY year_month;

-- 8. Brand performance over time
SELECT brand, year_month, listing_count,
       ROUND(avg_price, 0) AS avg_price
FROM brand_monthly_forecasting_data
WHERE brand IN ('Maruti', 'Hyundai', 'Honda', 'Toyota', 'Tata')
ORDER BY brand, year_month;

-- 9. Feature correlation analysis - high mileage vs price
SELECT mileage_bucket,
       COUNT(*) AS count,
       ROUND(AVG(selling_price), 0) AS avg_price,
       ROUND(AVG(car_age), 1) AS avg_car_age
FROM ml_features
GROUP BY mileage_bucket;

-- 10. Price segments distribution
SELECT price_segment,
       COUNT(*) AS count,
       ROUND(AVG(km_driven), 0) AS avg_km,
       ROUND(AVG(car_age), 1) AS avg_age
FROM ml_features
GROUP BY price_segment;
