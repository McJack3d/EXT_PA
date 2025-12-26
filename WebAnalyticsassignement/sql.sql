-- Web Analytics Assignment (Demo dataset): California Housing
-- Problem 3: “Coastal premium” after controlling for income (Jan 2021 snapshot framing)
--
-- Recommended engine for VS Code: DuckDB (runs SQL directly on CSV, no DB setup).
-- Run (option A): duckdb
--   $ duckdb
--   D .read WebAnalyticsassignement/sql.sql
--
-- If you use a VS Code SQL extension, set the connection to DuckDB.

-- 0) Load the CSV into a view (DuckDB reads CSVs natively)
CREATE OR REPLACE VIEW housing_train AS
SELECT *
FROM read_csv_auto('california_housing_train.csv', header = TRUE);

-- 1) Core extraction query (for statistical testing)
-- Control for income by comparing within income quintiles.
-- “Coastal-like” is approximated by being further West (more negative longitude).
WITH labeled AS (
	SELECT
		longitude,
		latitude,
		median_income,
		median_house_value,
		CASE
			WHEN longitude <= -121.0 THEN 'coastal_like'
			ELSE 'inland_like'
		END AS geo_group
	FROM housing_train
	WHERE median_income IS NOT NULL
		AND median_house_value IS NOT NULL
),
banded AS (
	SELECT
		*,
		NTILE(5) OVER (ORDER BY median_income) AS income_quintile
	FROM labeled
)
SELECT
	income_quintile,
	geo_group,
	median_house_value
FROM banded
WHERE income_quintile IN (3, 4)
ORDER BY income_quintile, geo_group;

-- 2) Summary table (useful for write-up: sample sizes + group means)
WITH labeled AS (
	SELECT
		median_income,
		median_house_value,
		CASE
			WHEN longitude <= -121.0 THEN 'coastal_like'
			ELSE 'inland_like'
		END AS geo_group
	FROM housing_train
	WHERE median_income IS NOT NULL
		AND median_house_value IS NOT NULL
),
banded AS (
	SELECT
		*,
		NTILE(5) OVER (ORDER BY median_income) AS income_quintile
	FROM labeled
)
SELECT
	income_quintile,
	geo_group,
	COUNT(*) AS n,
	AVG(median_house_value) AS avg_value,
	MEDIAN(median_house_value) AS median_value
FROM banded
GROUP BY income_quintile, geo_group
ORDER BY income_quintile, geo_group;

