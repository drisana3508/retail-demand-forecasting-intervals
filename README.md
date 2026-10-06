# retail-demand-forecasting-intervals
# Retail Demand Forecasting with Prediction Intervals

## Project overview

Retail demand is uncertain. A single forecast can hide the risk of ordering too much or too little inventory. This project builds a model that predicts a range of likely daily sales for each store-item pair and compares inventory decisions based on different demand quantiles.

The project uses the **Store Item Demand Forecasting Challenge** dataset from Kaggle. It forecasts sales one day ahead for each store-item pair using historical sales and calendar features.

The model produces q10, q50, and q90 forecasts:

- **q10** is the lower 10th-percentile forecast.
- **q50** is the median forecast.
- **q90** is the upper 90th-percentile forecast.

The q10–q90 range is intended to contain about 80% of actual sales. A separate calibration period is used to adjust the interval before evaluating it on the final test period.

## Dataset

Download the data from the [Kaggle Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only).

1. Sign in to Kaggle and accept the competition rules if prompted.
2. Open the competition’s **Data** tab and download `train.csv`.
3. The notebook also accepts the downloaded `train.csv.zip` file and extracts it.

The dataset contains daily sales for 10 stores and 50 items. The columns used are:

- `date`: sales date
- `store`: store identifier
- `item`: item identifier
- `sales`: units sold

The dataset does not include actual promotion, price, inventory availability, or holiday-event fields. The model therefore cannot directly use those signals to anticipate promotion-driven demand spikes.

Do not add Kaggle data files to this repository unless the dataset’s rules explicitly allow redistribution. The notebook asks the user to upload the data when it runs.

## Open and run the notebook

The notebook is `Copy_of_demand_forecasting.ipynb`.

1. Open the notebook file in this repository.
2. Select **Open in Colab** if the link is available, or download the notebook and upload it to [Google Colab](https://colab.research.google.com/).
3. In Colab, use **Runtime → Run all**.
4. When prompted, upload `train.csv.zip` downloaded from Kaggle. The notebook extracts `train.csv`.
5. If LightGBM is not already available in the runtime, run the notebook’s installation cell:

   ```python
   %pip install -q lightgbm
   ```

The notebook installs or imports the required tools, loads the data, engineers features, fits the models, calibrates intervals, evaluates accuracy and inventory policies, and plots the results.

## Modeling approach

### Time-based features

For each store-item series, the notebook creates:

- Sales lags of 1, 7, 14, and 28 days
- 7-day and 28-day rolling means
- 7-day and 28-day rolling standard deviations
- Day of week, day of month, month, week of year, and weekend indicator

Rolling statistics are shifted by one day before calculation. This ensures that a forecast for a given date does not use that date’s actual sales as an input.

The model uses a **global model**: one model learns across all stores and items, with store and item identifiers included as categorical features. This allows the model to share patterns across series while learning differences between stores and products.

### Quantile regression

Three LightGBM models are trained with quantile loss at q10, q50, and q90. Quantile loss penalizes underprediction and overprediction asymmetrically, allowing each model to estimate a different point in the demand distribution.

The quantiles are trained separately, so their predictions can occasionally cross. The notebook sorts the predicted values by row to keep the reported q10, q50, and q90 values in order.

### Chronological evaluation and calibration

The data is split by date, not randomly:

- **Training:** 2013-01-29 to 2017-11-05
- **Calibration:** 2017-11-06 to 2017-12-03
- **Test:** 2017-12-04 to 2017-12-31

The training period fits the model. The calibration period estimates a correction to the q10–q90 interval. The final test period measures performance on later, held-out dates.

This is a repeated **one-day-ahead** backtest: each day is forecast using sales history available before that day. As the backtest moves forward, actual sales from earlier test days can become historical lag features. This represents daily reforecasting; it is not a single 28-day forecast made at the beginning of the test period.

## Evaluation

The project reports both point accuracy and interval performance:

- **MAE:** average absolute error of the q50 forecast
- **RMSE:** square root of the average squared q50 error; larger errors receive more weight
- **Coverage:** fraction of actual sales inside the prediction interval
- **Interval width:** average distance between the lower and upper interval bounds
- **Inventory cost:** simulated shortage and leftover costs under alternative order policies

Coverage must be considered together with interval width. Making intervals excessively wide can increase coverage while making them less useful for inventory planning.

## Results

On the held-out test period:

| Measure | Result |
|---|---:|
| q50 MAE | 5.439 units |
| q50 RMSE | 7.081 units |
| q10–q90 coverage before calibration | 78.5% |
| q10–q90 coverage after calibration | 79.8% |
| Target q10–q90 coverage | 80% |
| Mean interval width after calibration | 17.25 units |

The calibration period produced an interval correction of approximately 0.23 units. Applying this correction to the test intervals increased measured coverage from 78.5% to 79.8%, close to the nominal 80% target.

Coverage on the highest-demand 10% of actual test days was 73.1%, below the overall coverage. By contrast, coverage across forecast-demand bands ranged from 78.4% to 81.1%. This suggests the model is particularly vulnerable to demand spikes it did not anticipate.

The dataset has no actual promotion, price, inventory, or event indicators. As a result, the model has limited information for predicting sudden spikes. The results should not be interpreted as evidence that promotions are handled reliably.

## Inventory decision and cost comparison

An inventory policy should reflect the cost of shortages relative to excess stock. In a one-period newsvendor setting, the cost-optimal demand quantile is:

```text
critical fractile = shortage cost / (shortage cost + leftover cost)
```

For the example assumptions used here:

- Shortage cost: 4 per unit
- Leftover cost: 1 per unit
- Cost-optimal quantile: 4 / (4 + 1) = q80

The test-period comparison was:

| Order policy | Mean cost per item-day | Mean shortage units | Mean leftover units |
|---|---:|---:|---:|
| q50 | 14.679 | 3.080 | 2.359 |
| q80 | 9.806 | 0.982 | 5.878 |
| q90 | 10.138 | 0.455 | 8.317 |

Under these illustrative costs, q80 had the lowest measured mean cost. q90 had fewer shortages, but more leftover units. These costs are examples, not retailer-specific estimates; they should be replaced with business costs such as lost margin, expedited replenishment, salvage value, and service-level penalties.

The q80 model is a decision forecast for the specified cost ratio. The q10–q90 interval is a separate uncertainty estimate.

## Reliability findings and limitations

- Overall calibrated test coverage was close to its 80% target.
- Coverage on the highest-demand actual days was lower, indicating risk around spikes.
- Forecast-demand bands had relatively similar coverage, but this does not rule out errors on unexpected events.
- The dataset does not provide real promotion or holiday flags, prices, or stock availability.
- Sales may not equal true demand if products were out of stock.
- The test period covers only the final 28 days in this historical dataset.
- Calibration performance can change when demand patterns shift.
- Inventory costs use illustrative assumptions and should be replaced before operational use.
- The model is evaluated for daily reforecasting with a one-day horizon. Multi-day lead-time decisions require forecasts of cumulative demand over the lead time.

## Potential improvements

1. Add genuine future-known price, promotion, holiday, and event data.
2. Evaluate with several rolling-origin validation periods, not only one final test block.
3. Compare against seasonal-naive forecasts such as sales from the same weekday one week earlier.
4. Measure coverage and interval width by product, store, weekday, forecast horizon, and demand level.
5. Add out-of-stock indicators or censored-demand treatment if inventory availability data is available.
6. Extend to M5 for event, SNAP, and price features, while using validation that matches the intended forecast horizon.
7. Replace the illustrative inventory cost assumptions with actual business economics.

## Reproducibility

The notebook uses a fixed random seed for LightGBM and contains the data preparation, training, calibration, evaluation, and inventory-cost calculations. To reproduce the results, download the Kaggle data, upload it when prompted in Colab, and run the notebook from top to bottom.

The measured values in this README come from one run on the stated chronological split. Re-running with changed code, packages, data, or split dates may produce different results.
### Seasonal-naive baseline comparison

The seasonal-naive baseline predicts each store-item’s sales using its sales from the same weekday one week earlier (`lag_7`). Its residual quantiles are estimated on the calibration period and applied to the test period.

| Measure | Seasonal-naive baseline | LightGBM |
|---|---:|---:|
| q50 MAE | 8.742 | 5.439 |
| q50 RMSE | 11.803 | 7.081 |
| 80% interval test coverage | 85.5% | 79.8% |
| Mean interval width | 30.64 units | 17.25 units |
| q80 inventory cost per item-day | 14.976 | 9.806 |

LightGBM improved point accuracy and had lower measured q80 inventory cost under the example shortage cost of 4 per unit and leftover cost of 1 per unit. The seasonal-naive residual interval covered more than its 80% nominal target, but was substantially wider. This comparison suggests LightGBM provides a more useful balance of coverage and interval width on this test period.
## M5 extension: results on a 300-series sample

The M5 experiment extends the Store Item project with calendar events, state-specific SNAP indicators, sell prices, and the same lag and rolling-sales features.

To keep the experiment manageable in Colab, the sample contains 300 item-store series: 100 each from CA_1, TX_1, and WI_1. The forecasts are repeated one day ahead, using only sales available before each forecast date. Results are from three consecutive 28-day test folds.

The 80% prediction interval is formed from q10 and q90 forecasts. Inventory cost assumes a shortage cost of 4 per unit and a leftover cost of 1 per unit.

| Test fold | q50 MAE: naive / LightGBM | q50 RMSE: naive / LightGBM | Interval coverage | Mean interval width | Event-day coverage | High-demand coverage | q80 cost: naive / LightGBM |
|---|---:|---:|---:|---:|---:|---:|---:|
| d_1858–d_1885, event period | 1.074 / 0.794 | 2.243 / 1.749 | 90.8% | 2.564 | 91.6% | 73.9% | 2.430 / 1.856 |
| d_1886–d_1913, recent period | 1.094 / 0.808 | 2.136 / 1.678 | 90.8% | 2.626 | No event days | 71.1% | 2.457 / 1.861 |
| d_1914–d_1941, final holdout | 1.153 / 0.836 | 2.169 / 1.606 | 89.8% | 2.648 | 89.2% | 65.2% | 2.512 / 1.909 |

LightGBM had lower q50 MAE, q50 RMSE, and q80 inventory cost than seasonal naive in all three folds. Its q10–q90 intervals covered about 90% of test rows, above their nominal 80% target, so they were conservative overall. Coverage was lower on the high-demand subset, showing that the intervals still miss many demand spikes. The calibration correction was zero in each fold.

High-demand rows are defined using each series’ 90th-percentile test demand, with ties included; this can make the subset larger than 10% of rows. The event fold includes St. Patrick’s Day, Purim, and Easter. The final holdout includes Pesach End, Orthodox Easter, Cinco de Mayo, and Mother’s Day.

These are sample-based results, not full M5 competition results. The sample includes three stores and 300 series, and the inventory costs are illustrative. The dataset does not provide a direct promotion flag; event, SNAP, and price features are used as available demand signals.
### Run the M5 extension

The M5 experiment is in [`demand_forecasting_M5.ipynb`](demand_forecasting_M5.ipynb).

1. Open the [M5 Forecasting – Accuracy Data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data), sign in, and accept the competition rules if prompted.
2. Click **Download All** to download the M5 ZIP archive. It includes `sales_train_validation.csv`, `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv`.
3. Open `demand_forecasting_M5.ipynb` in Google Colab and choose **Runtime → Run all**.
4. When prompted by the notebook’s upload cell, select the M5 ZIP archive. The notebook extracts the files into `/content/m5_data`.
5. If LightGBM is not available in the Colab runtime, run the notebook’s installation cell:

   ```python
   %pip install -q lightgbm
   ```

The notebook samples 100 item-store series from each of CA_1, TX_1, and WI_1, then runs three chronological 28-day backtests. It uses the M5 calendar, SNAP, and sell-price data. The Kaggle CSV and ZIP files are not included in this repository; download them from Kaggle when running the notebook.
