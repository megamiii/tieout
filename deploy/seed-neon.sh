#!/usr/bin/env sh
# Copy the local tieout database (docker compose) to a remote Postgres, such as
# a free Neon project. Safe to re-run: existing tieout objects are replaced.
#
#   NEON_DATABASE_URL='postgresql://USER:PASSWORD@HOST/neondb?sslmode=require' sh deploy/seed-neon.sh
#
# pg_dump and psql run inside the local database container, so no Postgres
# client is needed on the host. The URL is passed as an environment variable
# rather than an argument, so it doesn't appear in the process list.
set -eu
: "${NEON_DATABASE_URL:?set NEON_DATABASE_URL to the target connection string}"

docker compose exec -T db pg_dump -U tieout -d tieout \
    --no-owner --no-privileges --clean --if-exists \
  | docker compose exec -T -e TARGET_URL="$NEON_DATABASE_URL" db \
    sh -c 'psql "$TARGET_URL" --quiet --set ON_ERROR_STOP=1 > /dev/null'

# What arrived, and how much of the free tier's 0.5 GB it uses.
docker compose exec -T -e TARGET_URL="$NEON_DATABASE_URL" db sh -c 'psql "$TARGET_URL" --command "
  SELECT (SELECT count(*) FROM chunks) AS chunks,
         (SELECT count(*) FROM xbrl_facts) AS xbrl_facts,
         (SELECT count(*) FROM agent_runs) AS agent_runs,
         (SELECT count(*) FROM workflow_runs) AS workflow_runs,
         pg_size_pretty(pg_database_size(current_database())) AS size"'
echo "Seeded."
