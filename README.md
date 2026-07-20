# FlexiLab Movement Intelligence API — V100

FastAPI backend for Client, Trainer and Operator products.

## Processes

Web API:

```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```

Analysis worker:

```bash
python worker.py
```

Corporate import worker:

```bash
python operator_worker.py
```

## Deployment

Apply `migrations/20260720_v100_operator_security.sql` before deploying V100.
Then replace the UUID placeholder in `migrations/BOOTSTRAP_FIRST_OPERATOR.sql`
to create the first `super_admin` Operator.

Environment variable names are documented in `env.example`. Never commit private
Supabase, Stripe, email or webhook secrets.

## Validation

```bash
python -m py_compile app.py account_api.py stripe_api.py trainer_api.py operator_api.py worker.py operator_worker.py prescription_engine.py program_engine.py screening_access.py
python -m unittest discover -s tests -v
```
