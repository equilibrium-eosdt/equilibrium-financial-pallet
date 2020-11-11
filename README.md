# Financial Pallet

_This project is currently under active development._

## Overview

Equilibrium's financial pallet is an open-source substrate module that subscribes to external price feed/oracle, gathers asset prices and calculates financial metrics based on the information collected.  

## Documentation

For the detailed documentation please refer to [docs/FINANCIAL.md](docs/FINANCIAL.md).

## Installation

Make sure you have done all steps described in [Installation page](https://substrate.dev/docs/en/knowledgebase/getting-started/) of the Substrate Developer Hub.

To build project run:

```bash
cargo build
```

## Tests

To run unit tests type:

```bash
cargo test
```

You can see unit test report [here](docs/test-report.txt).

In case you want run code coverage tool, please follow [instructions](https://github.com/xd009642/tarpaulin#installation) to install tarpaulin.

To create code coverage report run:

```bash
cargo tarpaulin -v
```

You can see code coverage report [here](docs/tarpaulin-report.html).

To run integration tests you need purge and run node first (see the next section for more information).

Then type:

```bash
cargo run -p integration-testing
```

Note that integration tests take several minutes to complete.

## Running the Node

First of all please ensure that your development chain's state is empty:

```bash
cargo run --bin node-template purge-chain --dev
```

Now you can start the development chain:

```bash
cargo run --bin node-template --dev
```

### Connecting Explorer

It can be very useful to connect UI to the node you just started.

To do this open https://polkadot.js.org/apps/#/explorer in your browser first.

Follow these steps to register required custom types:

* In the main menu choose Settings tab;
* In the Settings submenu choose Developer tab;
* Copy content of the [custom-types.json](custom-types.json) file into text box on the page;
* Press Save button.

## Development Roadmap

| Milestone # | Description |
| --- | --- |
| 1 | Initial implementation, subscription to oracle |
| 2 | Returns, volatilities, and correlation calculations |
| 3 | Portfolio volatility calculations, Value at Risk calculations |

## Future Plans

The Financial pallet will be further used in Equilibrium’s DeFi parachain for the risk and pricing models to calculate interest rate fees, assess borrower portfolio risks and overall system risks. 

We have a plan of the further financial pallet evolution that will add complex and non-parametric ways of volatility, correlation and portfolio risk estimations. We will offload heavy calculations in some of these methods to designated off-chain worker.

Following additional methodologies/calculations will be introduced: EWMA, GARCH, Garman-Klass volatility estimates, as well as GARCH, CVaR, Monte-Carlo simulation, Historical simulation, and Copula methods for assessing value at risk. 


## License

Equilibrium Financial Pallet is [Apache 2.0 licensed](LICENSE).
