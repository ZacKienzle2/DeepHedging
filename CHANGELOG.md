# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Entries are generated from [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)
by [git-cliff](https://git-cliff.org).

## [Unreleased]

### Added

- **deephedging:** Add the paper extensions and profiling fixes (#22) ([61b7715](https://github.com/ZacKienzle2/DeepHedging/commit/61b77157151c7f1cbf4d6c1e54fe8309449d2f70))

- **market:** Add a fused Merton jump-diffusion CUDA kernel (#12) ([5e52142](https://github.com/ZacKienzle2/DeepHedging/commit/5e5214289d38feb7c3c16928431939ad1889d13b))

- **frictions:** Add power-law impact and directional bid-ask costs (#10) ([3dbc4f7](https://github.com/ZacKienzle2/DeepHedging/commit/3dbc4f72e5b6132dc82ac45e6f502637e8f5e13b))

- **instruments:** Add asian average and lookback payoffs (#9) ([5b989f7](https://github.com/ZacKienzle2/DeepHedging/commit/5b989f70bdd58344b819d63f5354135484340335))

- **evaluation:** Add autodiff greeks and bootstrap inference (#6) ([b47ade1](https://github.com/ZacKienzle2/DeepHedging/commit/b47ade1430e7f503592cf81eb09a31eec63952c0))

- **risk:** Add spectral and mean-variance objectives (#3) ([954f448](https://github.com/ZacKienzle2/DeepHedging/commit/954f4482e1fa13bff0c10b442b05ec1adf62d57e))

- **training:** Add lr schedule and gradient clipping (#4) ([f46acbd](https://github.com/ZacKienzle2/DeepHedging/commit/f46acbd3977ae87aa11890c4ff4b101216e704fd))

- **bench:** Measure generated training and arm the regression gate ([ced1424](https://github.com/ZacKienzle2/DeepHedging/commit/ced14247407f78468f9693248a43718824a3dbb6))

- **market:** Extend in-graph generation to the two-asset market ([7b8822a](https://github.com/ZacKienzle2/DeepHedging/commit/7b8822a3e03a990662255af7df06f25dea67bf05))

- **training:** Generate batches inside the captured graph ([a5f1ce8](https://github.com/ZacKienzle2/DeepHedging/commit/a5f1ce8f4e8d28ef4a3b579ea4f8eadc6da10421))

- **training:** Regenerate paths from their noise streams in the backward ([ce63b27](https://github.com/ZacKienzle2/DeepHedging/commit/ce63b27b23d277d47a7192298bb09b119c0143f7))

- **experiments:** Offer the fused Heston sampler to the volatility studies ([36c6636](https://github.com/ZacKienzle2/DeepHedging/commit/36c663653d5100804c0855e13d0d3c73b14aa342))

- **frictions:** Add per-asset proportional cost rates ([dcfa4e9](https://github.com/ZacKienzle2/DeepHedging/commit/dcfa4e99c5c1d907c4170b988a1a8f546d83555a))

- **experiments:** Add architecture and objective studies ([1e1c022](https://github.com/ZacKienzle2/DeepHedging/commit/1e1c022e1bb6b4a3b554138af04f54222713f320))

- **policies:** Add no-transaction-band policy ([8192ceb](https://github.com/ZacKienzle2/DeepHedging/commit/8192ceb7a481ab50a0e0a596096806ed4bf97c8e))

- **experiments:** Add instrument span and BSDE dimension studies ([d6b6a33](https://github.com/ZacKienzle2/DeepHedging/commit/d6b6a333c57d6b9f17d3608410d4cc73d8fe848d))

- **experiments:** Test no-trade band against the one-third power law ([9166ec0](https://github.com/ZacKienzle2/DeepHedging/commit/9166ec0dff8d5e85919023a20ba0085c197d568a))

- **market:** Add tradable variance swap on Heston paths ([79c24a8](https://github.com/ZacKienzle2/DeepHedging/commit/79c24a8a6fdb0bf9c9b1c39c6c5721c660ddc731))

- **experiments:** Add barrier option hedging study ([d75c22c](https://github.com/ZacKienzle2/DeepHedging/commit/d75c22c9184bbf5a67f81b0b7b3ebbfd0abd037d))

- **experiments:** Record no-trade band probe in frontier runs ([2180f89](https://github.com/ZacKienzle2/DeepHedging/commit/2180f89a44a5dbb4391f2bd0b61cde0ae84ed39e))

- **experiments:** Add frontier and observability experiment runners ([ff11e2a](https://github.com/ZacKienzle2/DeepHedging/commit/ff11e2abf09867af4b3dfab1d8a7625321ec6032))

- Add a realised-volatility observation for partial information ([25fbd2d](https://github.com/ZacKienzle2/DeepHedging/commit/25fbd2da7a5b779da178f37a2cc55e66693641e0))

- Add american put valuation by least-squares monte carlo ([59c0fc7](https://github.com/ZacKienzle2/DeepHedging/commit/59c0fc76065fd9d7bb1526cc46a7c8f916d3d2db))

- Add pricing-route benchmark rows and a nightly regression gate ([737eb1e](https://github.com/ZacKienzle2/DeepHedging/commit/737eb1ee74716112a60cb1ae46ce8b6144e0ee8d))

- Add an append-only experiment record store ([057de8b](https://github.com/ZacKienzle2/DeepHedging/commit/057de8b3fd7404d73b19a27657adb0b37b0b501c))

- Combine graph capture with gradient checkpointing ([10489d1](https://github.com/ZacKienzle2/DeepHedging/commit/10489d1207321c677edbef8e589d9ac82fc91737))

- Add exponential tilting for tail importance sampling ([fa900e8](https://github.com/ZacKienzle2/DeepHedging/commit/fa900e8bb5be872dd49d48be62c94a08d4d5efb5))

- Add dupire local volatility closing the calibration loop ([9962ee1](https://github.com/ZacKienzle2/DeepHedging/commit/9962ee1342085faadb284a6417fc9bf10b3d3f1e))

- Fold path statistics in registers and price without the grid ([818e867](https://github.com/ZacKienzle2/DeepHedging/commit/818e867eb1634491f79f828dc3090c2e460f90e3))

- Add merton jump-diffusion with closed-form golden ([32e5680](https://github.com/ZacKienzle2/DeepHedging/commit/32e5680361e2b8e8dba0cee8fb709bb1a5768f39))

- Add vega-weighted multi-maturity heston calibration ([7d60ce4](https://github.com/ZacKienzle2/DeepHedging/commit/7d60ce4397966d715270f99469ed34e0918126b6))

- Add cos heston pricer and implied volatility inversion ([0798749](https://github.com/ZacKienzle2/DeepHedging/commit/0798749fd14ccb7b1cd39cc712b0cdf1985076a6))

- Hedge multi-asset books through the episode loop ([57dc989](https://github.com/ZacKienzle2/DeepHedging/commit/57dc9890fa17e1898c462a51bb33bc931462170d))

- Add correlated multi-asset simulation and basket payoffs ([82af401](https://github.com/ZacKienzle2/DeepHedging/commit/82af401230ec6cf376950fa6313246baae4cbb49))

- Carry auxiliary channels through episode graph capture ([d106e12](https://github.com/ZacKienzle2/DeepHedging/commit/d106e120a2dd53a27634ec61202cb78154eb095d))

- Capture the training iteration in a cuda graph ([a5c3808](https://github.com/ZacKienzle2/DeepHedging/commit/a5c38081c01999a3852d98663730de3cf2ac1b40))

- Add fused philox path generators ([3ce3c1b](https://github.com/ZacKienzle2/DeepHedging/commit/3ce3c1bb892fc2aadcdc38c450a296459bf7e8c6))

- Add throughput benchmark harness ([46d14f4](https://github.com/ZacKienzle2/DeepHedging/commit/46d14f415c7ca5f2ab23424ed94f313b9ca1c67a))

- Add optional policy compilation to the trainer ([9b65bc1](https://github.com/ZacKienzle2/DeepHedging/commit/9b65bc1f8c08d188c100715e5f0e479609e85295))

- Add streaming path-functional folds ([d06607c](https://github.com/ZacKienzle2/DeepHedging/commit/d06607c1393e8f151a7b18a0c1b3575ef2929e07))

- Add pricing seam with uncertainty-carrying estimates ([991077c](https://github.com/ZacKienzle2/DeepHedging/commit/991077c6ee1d40e07e8a61beaf726de85ffc3b82))

- Introduce market-state and noise-spec simulator contracts [**breaking**] ([111a442](https://github.com/ZacKienzle2/DeepHedging/commit/111a4429d6cd527efd448c9ff55d382378c668ab))

- Add feature-map protocol for policy observations ([e52557f](https://github.com/ZacKienzle2/DeepHedging/commit/e52557ffc368fd5c6b4b43f207e2d4620e8a0f94))

- Add deep bsde solver for semilinear pricing pdes ([151f391](https://github.com/ZacKienzle2/DeepHedging/commit/151f391f22c1b2632ded8827466a576205e55211))

- Add heston simulator, barrier payoff, and ci workflow ([6f8873b](https://github.com/ZacKienzle2/DeepHedging/commit/6f8873b481c444826922f198c42b10d02296d280))

- Add pure-pytorch deep hedging mvp ([2293baf](https://github.com/ZacKienzle2/DeepHedging/commit/2293baf9a342bc3c4db3d9ea3b40b7281be7ecf4))


### Build

- **template:** Update to the mutants and vale fixes ([557241b](https://github.com/ZacKienzle2/DeepHedging/commit/557241bdc2a966a3f4d386264fb5965ee6906033))

- Add contributor and ci infrastructure (#5) ([43cdde4](https://github.com/ZacKienzle2/DeepHedging/commit/43cdde4078eca58cbd9474f2ca1c0c269a1ae1c6))

- Cache ci dependencies and split the notebook extra ([5419920](https://github.com/ZacKienzle2/DeepHedging/commit/5419920f6116e952e33f18ae65fdb0a524e688ee))


### Changed

- **experiments:** Share the runner mechanics through one harness ([5d740fc](https://github.com/ZacKienzle2/DeepHedging/commit/5d740fc13518b1075bdb0e45ff10e2b2662cc099))

- **features:** Drop the unused realised-variance observation ([af0738e](https://github.com/ZacKienzle2/DeepHedging/commit/af0738ebe9d447b277a2aa410d2e18b43a8ab1a8))

- **experiment:** Accept BSDE runs through the record factory ([de99aed](https://github.com/ZacKienzle2/DeepHedging/commit/de99aedebc56c3bae4d930abd49ae3f7391753d9))

- **experiments:** Name runners after their studies ([85940cf](https://github.com/ZacKienzle2/DeepHedging/commit/85940cf9dc77aa9ab777dd60c65b2f5d205e8507))

- **experiments:** Drop observability ablation runner ([a5fe578](https://github.com/ZacKienzle2/DeepHedging/commit/a5fe5783a31038048f412d13103b7bdd4dc91dc0))

- Type the graphed callable return ([0cf8730](https://github.com/ZacKienzle2/DeepHedging/commit/0cf87300b2227604e1c9d78820599f8390e4a223))

- Use the capitalized environment variable lookup ([1e7d6ae](https://github.com/ZacKienzle2/DeepHedging/commit/1e7d6aeaf88a8646882c25e2a7826a3eae4c682b))


### Documentation

- **style:** Use plain prose in docstrings and documentation (#11) ([e0d24cb](https://github.com/ZacKienzle2/DeepHedging/commit/e0d24cbbd60000d273309ad22b290767d3486ff2))

- **roadmap:** Add roadmap to a state-of-the-art platform (#2) ([1cfdbac](https://github.com/ZacKienzle2/DeepHedging/commit/1cfdbac11068c2fe97ec32225ad8d6bc6e0328f8))

- **readme:** Surface the headline results and the actual scope ([7f5b83d](https://github.com/ZacKienzle2/DeepHedging/commit/7f5b83da933e4f7fbaa3e50a9e900e4c4e452c3c))

- **notebook:** Execute walkthrough with the new sections ([bcf560d](https://github.com/ZacKienzle2/DeepHedging/commit/bcf560ded3ae7f118ea5b34b2613c3860b288fe2))

- **notebook:** Demonstrate band probe, variance swap, and barrier hedger ([2ce5d11](https://github.com/ZacKienzle2/DeepHedging/commit/2ce5d110b09858b13f31340417d7845e4cf986c7))

- Extend the walkthrough to the full model and pricing surface ([00d9bab](https://github.com/ZacKienzle2/DeepHedging/commit/00d9babd07a5feffd690f86c44d49f26fac130cd))

- Document raised exceptions across public entry points ([94d512c](https://github.com/ZacKienzle2/DeepHedging/commit/94d512cc85411803299d02f22de5b4951feebb2c))

- Refresh readme for the current module layout ([81d3bbe](https://github.com/ZacKienzle2/DeepHedging/commit/81d3bbe33f5435c6f3c3dc975ddb2f6a06a31712))

- Clarify noise-stream composition and seeding rules ([c88a6f8](https://github.com/ZacKienzle2/DeepHedging/commit/c88a6f89031be76db8fb6c74b20f86db3c00512b))

- Add executable walkthrough notebook ([66ebd42](https://github.com/ZacKienzle2/DeepHedging/commit/66ebd4242e2682e3f5311a415fabd7127030f9e0))


### Fixed

- **training:** Tighten the capture headroom warning on Windows ([5e1ce64](https://github.com/ZacKienzle2/DeepHedging/commit/5e1ce64052ccdbd5ef713003cd582a70daf70ae9))

- **ci:** Make the GPU lane fail loudly instead of skipping silently ([f578668](https://github.com/ZacKienzle2/DeepHedging/commit/f5786685e376a78f045b44dc9a644216c730fa65))

- **experiments:** Destroy graphs before releasing the cache between runs ([3b9c2de](https://github.com/ZacKienzle2/DeepHedging/commit/3b9c2de8f3a63b05fab119dbe56fd6617da44acc))

- **experiments:** Free capture pools between runs and double budgets ([d9f46fc](https://github.com/ZacKienzle2/DeepHedging/commit/d9f46fc74c559c9a1fa310484d42569f1e44fb7c))

- **experiments:** Hedge daily over thirty days in the frontier ([526f10b](https://github.com/ZacKienzle2/DeepHedging/commit/526f10bed71650ae7dafbc35fbe5ac0ab24176ee))

- Keep importance weights finite and weight the eval estimator ([4468abb](https://github.com/ZacKienzle2/DeepHedging/commit/4468abb763037c88ac9ad6cfc7afe6ac48d12a95))

- Harden the benchmark harness methodology ([9456790](https://github.com/ZacKienzle2/DeepHedging/commit/94567903cbf141e22791a5edc0cd2006e6017908))

- Warn when graph capture nears device memory ([a49eab5](https://github.com/ZacKienzle2/DeepHedging/commit/a49eab5c01b8fedc907d0653d64d10b89bbb05eb))

- Harden dupire inputs and tighten the fold pricing route ([49f62be](https://github.com/ZacKienzle2/DeepHedging/commit/49f62be8ac3284fdea2d3d9641ef6f690271bd6f))

- Use the non-uniform three-point time stencil in dupire ([6382626](https://github.com/ZacKienzle2/DeepHedging/commit/638262609b951eb6f7b7920a66611c62f8079735))

- Discount monte carlo prices and guard the pricing measure ([a1e4b5b](https://github.com/ZacKienzle2/DeepHedging/commit/a1e4b5b50a5f13844498c018ea1be04c063085b5))

- Clamp jump counts at the truncation bound ([41f778d](https://github.com/ZacKienzle2/DeepHedging/commit/41f778d5b43baec4226f80d56d8ec680588800b0))

- Guard the analytic pricer drift and widen the ci lanes ([f7896f4](https://github.com/ZacKienzle2/DeepHedging/commit/f7896f4479e3e55ca8940704f60277ee277fc342))

- Harden calibration inputs and bracket the vol inversion ([293a808](https://github.com/ZacKienzle2/DeepHedging/commit/293a808eca9cd843dbd9b5d1b6ae9680f7b36fb8))

- Harden multi-asset tests and shape contracts ([ce05bab](https://github.com/ZacKienzle2/DeepHedging/commit/ce05bab97da339bc62968be107a7ebed6d2da4b1))

- Bound the kernel step count before narrowing ([a4db06f](https://github.com/ZacKienzle2/DeepHedging/commit/a4db06f34cc4be7dbeb9ab441bf9af9ae4eb5e75))

- Move the risk measure to the policy device in the trainer ([1f9d048](https://github.com/ZacKienzle2/DeepHedging/commit/1f9d0483d27cb3abc3d6bf57f377cb6e169f9317))

- Harden estimators and tighten hot paths ([8a1a7ae](https://github.com/ZacKienzle2/DeepHedging/commit/8a1a7ae56753dddf6c3f6a10e0c67513a135be3d))


### Maintenance

- **experiments:** Record full study results ([a688868](https://github.com/ZacKienzle2/DeepHedging/commit/a68886883ab8ea2b605edc255cbc705afe191242))

- Scaffold repository ([73a3ecf](https://github.com/ZacKienzle2/DeepHedging/commit/73a3ecf2bbcba9ab1d44b0c345c93286f47d8fad))


### Performance

- **frictions:** Cache the per-asset rate tensor by device and dtype ([5bc94a3](https://github.com/ZacKienzle2/DeepHedging/commit/5bc94a3873a6a3105d1234e7d5605bd44dbb020e))

- Hoist per-step scalar work out of the heston loop ([180c53d](https://github.com/ZacKienzle2/DeepHedging/commit/180c53d2fe31bc4c5af747ab958ad6fe908a3692))

- Cache the log grid and add optional policy autocast ([84f66b9](https://github.com/ZacKienzle2/DeepHedging/commit/84f66b904652480c44df23b8bf1b792a31a59ca8))


### Tests

- Pin stressed swap positivity and the eval routing invariant ([57bc07f](https://github.com/ZacKienzle2/DeepHedging/commit/57bc07ff1581bd7dcbfb118b3762fb062f7f6f2c))

- Pin the antithetic and control-variate design decisions ([3e88cf5](https://github.com/ZacKienzle2/DeepHedging/commit/3e88cf5a58784e7ce35ad05bcc038b643b74d67a))

- Assert calibration fit in implied volatility space ([1df7e0b](https://github.com/ZacKienzle2/DeepHedging/commit/1df7e0b7ee38cad2bfba8ae29cda4aa11c8e195d))

- Size the calibration refit tolerance to the weighted objective ([4428ad7](https://github.com/ZacKienzle2/DeepHedging/commit/4428ad712a46cf00d37b0ce7ae28b5fc64830fc5))



