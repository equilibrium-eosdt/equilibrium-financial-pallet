# Financial pallet

## Overview

Equilibrium's financial pallet is an open-source substrate module that subscribes to external price feed/oracle, gathers asset prices and calculates financial metrics based on the information collected.  

## Data model

### Settings

This sections contains description of constants which should be set with the deploy of the financial pallet. They govern the behaviour and calculations of pallet functions as detailed further in the Functions section. 

#### Period

The period of the collected prices in minutes. Possible values are 1, 10, 60, 240, 1440, 10080 minutes.   
By default = 1440 minutes \(1 day\). 

```rust
PricePeriod: u32 = 1440;
```

#### Price Count

Number of price data points stored and used for calculations \(length of price vector for each asset\).   
Max = 180. By default = 30. 

```rust
PriceCount: u32 = 30; 
```

#### Return type

Indicates what returns will be used in calculations of volatilities, correlations, and value at risk: regular or log returns. 

The choice of return type also governs the method for Value at Risk \(VAR\) calculation: Regular type should be used when arithmetic returns are used and are assumed to be normally distributed, while log normal type should be used when geometric returns \(log returns\) are used and are assumed to be normally distributed. We suggest using the latter approach, as it doesn't lead to losses greater than a portfolio value unlike the normal VaR.  
  
Log returns are used by default. 

```rust
ReturnType: CalcReturnType = CalcReturnType::Log; 
```

#### Volatility type

Indicates the method for calculating volatility: regular or exponentially weighted.   
Regular type is a standard statistical approach of calculating standard deviation of returns using simple average, while exponentially weighted type gives more weight to most recent data given the decay value or period of exponentially weighted moving average.   
  
Regular type is used by default

```rust
VolatilityType: CalcVolatilityType = CalcVolatilityType::Regular;
```

#### Correlation type

Indicates the method for calculating correlations: regular or exponentially weighted. Regular type is a standard statistical approach of calculating Pearson correlation coefficient between two return series, while exponentially weighted type uses exponentially weighted moving average to give more weights to most recent observations of pair of returns.   
  
Note: correlations on lower periods are more susceptible to data noise and reflect actual dependencies between returns poorly. For robustness we recommend to use daily or weekly periods for correlations. 

Regular type is used by default. 

```rust
CorrelationType: CalcCorrelationType = CalcCorrelationType::Regular;
```

### Storage

This section describes the storage of the financial pallet and how the relevant data is organised within it. 

#### Prices

Double mapping from Asset and Period to the array of prices of given DataPoints length 

```rust
pub Prices get(fn prices): double_map hasher(blake2_128_concat) Asset, hasher(blake2_128_concat) PricePeriod => Vec<FixedNumber>;
```

#### LastUpdate

Double mapping from Asset and Period to the UNIX timestamp indicating when the last price data was received. 

```rust
pub LastUpdate get(fn last_update): double_map hasher(blake2_128_concat) Asset, hasher(blake2_128_concat) PricePeriod => u64;
```

#### Volatility

Asset volatility mapping from Asset to Volatility figure. Stores current asset volatility or 0 if not enough price points for calculation. 

```rust
pub Volatility get(fn volatility): map hasher(blake2_128_concat) Asset => FixedNumber;
```

#### Correlation

Stores pairwise asset correlation. Double mapping from Asset and Asset to Correlation figure.

```rust
pub Corellation get(fn corellation): double_map hasher(blake2_128_concat) Asset, hasher(blake2_128_concat) Asset => FixedNumber;
```

## Functions

### calc\_return

Calculates return vector for each asset given respective price array.

#### _Parameters_

* `return_type`: type of return to be used in calculation.
* `asset` : asset to calculate returns for. 

####  _Returns_

*  Vector of returns for specified `asset` or NotEnoughPoints if not enough data points to perform calculation. 

#### _Errors_

```rust
enum CalcReturnError {
    NotEnoughPoints,
}
```

#### Declaration

`fn calc_return(return_type: CalcReturnType, asset: Asset) -> Result<Vec<FixedNumber>, CalcReturnError>;`

#### _Function sequence_

1. if return\_type is normal for `asset` calculate `returns` vector: `returns[i] = Prices[i] / Prices[i-1] - 1;`
2. if return\_type is log normal for `asset` calculate `returns` vector: `returns[i] = ln (Prices[i] / Prices[i-1]);`
3. return `returns` vector

### calc\_vol

Calculates volatility for specified `asset` given it's `returns`

#### _Parameters_

* `vol_type`: type of volatility to be used in calculation
* `asset` : asset to calculate returns for. 
* `ewma_length`: applicable when `voltype` is ewma. Specifies the period of exponentially weighted moving average. by default = 36. 
* `return_type`: type of return to be used in calculation.

####  _Returns_

* FixedNumber volatility number for given `asset`or NotEnoughPoints if not enough data points to perform calculation. 

#### _Errors_

```rust
enum CalcVolError {
    NotEnoughPoints,
}
```

#### _Declaration_

`fn calc_vol(vol_type: CalcVolatilityType, asset: Asset, ewma_length: u32, return_type: CalcReturnType) -> Result<FixedNumber, CalcVolError>;`

#### _Function sequence_

1. if vol\_type is regular for given `asset` calculate `volatility`:  
   `returns = calc_return(return_type, asset)   
   sqrt ( 1 / [length (returns) - 1] * sum [(returns[i] - mean (returns)^2])`

2. if vol\_type is ewma for given `asset` calculate `volatility`: `price_diff = price - lag(price,1);  price_diff(1,:) = [];  decay = 2/(ewma_length+1);  variance(1) = price_diff(1).^2;  for k = 2:length(price_diff)      variance(k) = price_diff(k).^2 .* decay + variance(k-1) .* (1 - decay); end volatility = sqrt(variance)` 
3. return `volatility` number for each asset. 

### calc\_corr

Calculates pairwise correlations between `asset1` and `asset2`given their respective `returns`

#### _Parameters_

* `asset1`: first asset
* `asset2`: second asset
* `corr_type`: type of correlation to be used in calculation
* `ewma_length`: applicable when `vol_type` is ewma, specifies the period of exponentially weighted moving average. By default = 36. 
* `return_type`: type of return to be used in calculation.

####  _Returns_

* FixedNumber correlation number for a pair of assets or NotEnoughPoints if not enough data points to perform calculation. 

#### _Errors_

```rust
enum CalcCorrError {
    NotEnoughPoints,
}
```

#### _Declaration_

`fn calc_corr(asset1: Asset, asset2: Asset, corr_type: CalcCorrelationType, ewma_length: u32, return_type: CalcReturnType) -> Result<FixedNumber, CalcCorrError>;`

#### _Function sequence_

1. if corr\_type is regular calculate `correlation`:  
   `returns1 = calc_return(return_type, asset1)  
   returns2 = calc_return(return_type, asset2)  
   1 / [length (returns1) - 1] * sum [(returns1[i] - mean (returns1)  (returns2[i] - mean (returns2)]) / s(asset1) / s(asset2)`where s\(asset\) is a volatility calculated for given asset.

2. if corr\_type is ewma calculate `correlation`:  
   `[r,c] = size([return1, return2]);   
   data_mwb = return-repmat(mean(return,1),r,1); % de-mean returns   
   decay = 2/(ewmaLength + 1);   
   decayvec = decay.^(0:1:r-1)';   
   data_tilde = repmat(sqrt(decayvec),1,c) .* data_mwb;                                   
   cov_ewma = 1/sum(decayvec)(data_tilde'data_tilde);   
   correlation = zeros(c);` 

   `for i = 1:c   
       for j = 1:c   
           correlation(i,j) = cov_ewma(i,j)/sqrt(cov_ewma(i,i)*cov_ewma(j,j));            
       end   
   end`  

3. return `correlation` number for specified pair of assets. 

### calc\_portf\_vol

Calculates portfolio volatility for given `account_id` by evaluating its balances

#### _Parameters_

* `account_id`: account id 

####  _Returns_

* FixedNumber volatility number for portfolio of assets of given `account_id` or NotEnoughPoints if not enough data points to perform calculation. 

#### _Errors_

```rust
enum CalcPortfVolError {
    NotEnoughPoints,
}
```

#### _Declaration_

`fn calc_prtf_vol(account_id: AccountId) -> Result<FixedNumber, CalcPortfVolError>;`  


#### _Function sequence_

1. `volatility = sqrt (weights’ *`_`token_covariance *`_ `weights)`  where   `token_covariance` is an assets covariance matrix.  `weights` is a vector of relative cash weights of each asset on the `account_id` balance

### calc\_portf\_var

NB! polkadot's rust implementation might not have libraries to work with distributions and z-scores, so we provide a simple method of calculating value at risk where z-score number is an input to the method itself rather than a confidence value \(probability of return being z standard deviations from the mean\). 

#### _Parameters_

* `account_id`: account id to calculate portfolio VAR for
* `return_type`: type of return / VaR calculation normal or log normal
* `z_score` : no. of standard deviations to consider. 

####  _Returns_

* FixedNumber VAR number for portfolio of assets of given `account_id` or NotEnoughPoints if not enough data points to perform calculation. 

#### _Errors_

```rust
enum CalcPortfVarError {
    NotEnoughPoints,
}
```

#### _Declaration_

`fn calc_portf_var(account_id: AccountId, return_type: CalcReturnType, z_score: u32) -> Result<FixedNumber, CalcPortfVarError>;`

#### _Function sequence_

1. if return\_type is normal `VaR = -mean(weights' * returns) + volatility * z_score`   where   `weights` is a vector of relative cash weights of each asset on the `accountId` balance `returns`are corresponding returns for each asset on the balance of the `account_id`  `volatility` is a portfolio volatility \(calculated in `calc_portf_vol`function\)
2. if return\_type = 'log normal'  
   `VaR = 1 - exp [mean(weights' * returns) - volatility * z_score]`   


   where   
  
   `weights` is a vector of relative cash weights of each asset on the `account_id` balance  
   `returns`are corresponding returns for each asset on the balance of the `account_id`   
   `volatility` is a portfolio volatility \(calculated in `calc_portf_vol`function\)

## Subscriptions

The financial pallet will be subscribed to oracle price changes:

There is a Trait setting inside oracle:

```rust
#[impl_for_tuples(5)]
pub trait OnNewPrice {
fn on_new_price(currency: &Currency, price: FixedI64);
}
```

Financial Trait subscribes to Oracle Trait and receives price updates \(`on_new_price` events\) each time oracle gets a new price. Oracle itself may not have complete price history, all price history needed for calculations is stored within the financial pallet.  


## Math functions

We will use [https://github.com/encointer/substrate-fixed](https://github.com/encointer/substrate-fixed) library for square root, natural logarithm, square, exponent and power functions. 

