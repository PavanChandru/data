from pathlib import Path
import pandas as pd
import hashlib
import io
import time

start_time = time.perf_counter()

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "test_data"

REQUEST_FILE = DATA_DIR / "mock_db" / "request.csv"
USER_FILE = DATA_DIR / "mock_db" / "users.csv"
LOOKUP_DIR = DATA_DIR / "lookup_tables"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

results = []


def report(name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed))

    print(f"[{status}] {name}")

    if details:
        print(f"       {details}")


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


# --------------------------------------------------
# Load synthetic data
# --------------------------------------------------

print("\n========== LOADING SYNTHETIC DATA ==========\n")

requests = load_csv(REQUEST_FILE)
users = load_csv(USER_FILE)

lookup_files = {
    "request_for": "request_for.csv",
    "request_islead": "request_isleadvol.csv",
    "category": "help_categories.csv",
    "request_type": "request_type.csv",
    "priority": "request_priority.csv",
    "status": "request_status.csv",
}

lookups = {
    name: load_csv(LOOKUP_DIR / filename)
    for name, filename in lookup_files.items()
}

print(f"Users: {len(users)}")
print(f"Requests: {len(requests)}")

# --------------------------------------------------
# 1. Source data validation
# --------------------------------------------------

print("\n========== SOURCE DATA VALIDATION ==========\n")

report(
    "Request dataset is not empty",
    len(requests) > 0
)

report(
    "User dataset is not empty",
    len(users) > 0
)

report(
    "Request IDs are unique",
    requests["req_id"].is_unique
)

report(
    "User IDs are unique",
    users["user_id"].is_unique
)

# --------------------------------------------------
# 2. Foreign key validation
# --------------------------------------------------

print("\n========== FOREIGN KEY VALIDATION ==========\n")

foreign_keys = {
    "req_user_id": ("users", "user_id"),
    "req_for_id": ("request_for", "req_for_id"),
    "req_islead_id": ("request_islead", "req_islead_id"),
    "req_cat_id": ("category", "cat_id"),
    "req_type_id": ("request_type", "req_type_id"),
    "req_priority_id": ("priority", "req_priority_id"),
    "req_status_id": ("status", "req_status_id"),
}

for request_column, (lookup_name, lookup_column) in foreign_keys.items():

    valid_ids = set(lookups[lookup_name][lookup_column]) \
        if lookup_name != "users" else set(users[lookup_column])

    null_count = requests[request_column].isna().sum()

    unmatched_count = (
        ~requests[request_column].isin(valid_ids)
    ).sum()

    report(
        f"Foreign key: {request_column}",
        null_count == 0 and unmatched_count == 0,
        f"NULL={null_count}, unmatched={unmatched_count}"
    )

# --------------------------------------------------
# 3. Prepare request columns
# --------------------------------------------------

print("\n========== PREPARING ARCHIVE DATA ==========\n")

archive = requests.copy()

archive = archive.rename(columns={
    "req_user_id": "creator_id",
    "last_update_date": "last_updated_at",
})

# --------------------------------------------------
# 4. Perform six LEFT JOINs
# --------------------------------------------------

print("\n========== EXECUTING LOOKUP JOINS ==========\n")

join_config = [
    ("request_for", "req_for_id", "req_for"),
    ("request_islead", "req_islead_id", "req_islead"),
    ("category", "req_cat_id", "req_cat_name"),
    ("request_type", "req_type_id", "req_type"),
    ("priority", "req_priority_id", "req_priority"),
    ("status", "req_status_id", "req_status"),
]

for lookup_name, request_key, label_column in join_config:

    lookup = lookups[lookup_name]

    if lookup_name == "category":
        lookup = lookup.rename(
        columns={"cat_name": "req_cat_name"}
    )

    lookup_key = {
        "request_for": "req_for_id",
        "request_islead": "req_islead_id",
        "category": "cat_id",
        "request_type": "req_type_id",
        "priority": "req_priority_id",
        "status": "req_status_id",
    }[lookup_name]

    lookup = lookup[[lookup_key, label_column]]

    archive = archive.merge(
        lookup,
        how="left",
        left_on=request_key,
        right_on=lookup_key,
        validate="many_to_one"
    )

    if lookup_key != request_key:
        archive = archive.drop(columns=[lookup_key])

    report(
        f"LEFT JOIN: {lookup_name}",
        len(archive) == len(requests),
        f"Rows after join: {len(archive)}"
    )

# --------------------------------------------------
# 5. Validate joined output
# --------------------------------------------------

print("\n========== JOINED OUTPUT VALIDATION ==========\n")

report(
    "No requests lost during joins",
    len(archive) == len(requests),
    f"Source={len(requests)}, Output={len(archive)}"
)

report(
    "No duplicate request IDs after joins",
    archive["req_id"].is_unique
)

label_columns = [
    "req_for",
    "req_islead",
    "req_cat_name",
    "req_type",
    "req_priority",
    "req_status",
]

for column in label_columns:

    missing = archive[column].isna().sum()

    report(
        f"Readable labels populated: {column}",
        missing == 0,
        f"Missing labels={missing}"
    )

# --------------------------------------------------
# --------------------------------------------------
# 6. Test NULL and broken references
# --------------------------------------------------

print("\n========== EDGE CASE TESTING ==========\n")

edge_cases = {
    "request_for": {
        "request_column": "req_for_id",
        "lookup_key": "req_for_id",
        "label": "req_for",
    },
    "request_islead": {
        "request_column": "req_islead_id",
        "lookup_key": "req_islead_id",
        "label": "req_islead",
    },
    "category": {
        "request_column": "req_cat_id",
        "lookup_key": "cat_id",
        "label": "cat_name",
    },
    "request_type": {
        "request_column": "req_type_id",
        "lookup_key": "req_type_id",
        "label": "req_type",
    },
    "priority": {
        "request_column": "req_priority_id",
        "lookup_key": "req_priority_id",
        "label": "req_priority",
    },
    "status": {
        "request_column": "req_status_id",
        "lookup_key": "req_status_id",
        "label": "req_status",
    },
}

for lookup_name, config in edge_cases.items():

    request_column = config["request_column"]
    lookup_key = config["lookup_key"]
    label_column = config["label"]

    lookup = lookups[lookup_name][
        [lookup_key, label_column]
    ].copy()

    # Create two synthetic edge-case requests.
    test_requests = requests.iloc[[0, 1]].copy()

    # Case 1: NULL foreign key.
    test_requests.iloc[
        0, test_requests.columns.get_loc(request_column)
    ] = pd.NA

    # Case 2: Foreign key pointing to a nonexistent record.
    test_requests.iloc[
        1, test_requests.columns.get_loc(request_column)
    ] = -999999

    # LEFT JOIN should preserve both requests.
    joined = test_requests.merge(
        lookup,
        how="left",
        left_on=request_column,
        right_on=lookup_key,
        validate="many_to_one"
    )

    # Remove duplicate lookup key if necessary.
    if lookup_key != request_column:
        joined = joined.drop(columns=[lookup_key])

    # Both requests must survive.
    rows_preserved = len(joined) == len(test_requests)

    # Both unresolved references should have NULL labels.
    labels_null = joined[label_column].isna().all()

    report(
        f"{lookup_name}: NULL reference preserved",
        rows_preserved and labels_null
        and pd.isna(joined.iloc[0][label_column])
    )

    report(
        f"{lookup_name}: Unmatched reference preserved",
        rows_preserved and labels_null
        and pd.isna(joined.iloc[1][label_column])
    )

# --------------------------------------------------
# 7. Test duplicate detection
# --------------------------------------------------

print("\n========== DUPLICATE TESTING ==========\n")

duplicate_requests = pd.concat(
    [requests, requests.iloc[[0]]],
    ignore_index=True
)

report(
    "Duplicate request detection works",
    duplicate_requests["req_id"].duplicated().any()
)

duplicate_lookup = pd.concat(
    [
        lookups["status"],
        lookups["status"].iloc[[0]]
    ],
    ignore_index=True
)

report(
    "Duplicate lookup key detection works",
    duplicate_lookup["req_status_id"].duplicated().any()
)

# --------------------------------------------------
# 8. Deterministic output / rerun test
# --------------------------------------------------

print("\n========== RE-RUN TEST ==========\n")

def export_bytes(data):
    return data.to_csv(
        index=False,
        lineterminator="\n"
    ).encode("utf-8")


first_export = export_bytes(archive)
second_export = export_bytes(archive.copy())

first_hash = hashlib.sha256(first_export).hexdigest()
second_hash = hashlib.sha256(second_export).hexdigest()

report(
    "Repeated export produces identical content",
    first_hash == second_hash,
    f"SHA256={first_hash}"
)

# Save final synthetic archive.
output_file = OUTPUT_DIR / "synthetic_archive.csv"

output_file.write_bytes(first_export)

print(f"\nOutput saved: {output_file}")

# --------------------------------------------------
# 9. Timestamp parsing
# --------------------------------------------------

print("\n========== TIMESTAMP VALIDATION ==========\n")

for column in ["submission_date", "serviced_date", "last_update_date"]:

    if column not in requests.columns:
        continue

    parsed = pd.to_datetime(
        requests[column],
        errors="coerce"
    )

    invalid = (
        requests[column].notna()
        & parsed.isna()
    ).sum()

    report(
        f"Timestamp parseable: {column}",
        invalid == 0,
        f"Invalid timestamps={invalid}"
    )

print(
    "\nNOTE: Timestamp parsing does not establish UTC. "
    "UTC behavior requires a separate timezone test."
)

# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n========== FINAL TEST SUMMARY ==========\n")

passed = sum(1 for _, result in results if result)
failed = sum(1 for _, result in results if not result)

print(f"Total checks: {len(results)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")

if failed:
    print("\nSome checks failed. Review the output above.")
else:
    print("\nAll synthetic verification checks passed.")

elapsed_time = time.perf_counter() - start_time

print(f"\nTotal execution time: {elapsed_time:.2f} seconds")