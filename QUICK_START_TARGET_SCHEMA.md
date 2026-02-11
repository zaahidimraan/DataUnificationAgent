# Quick Start Guide - Target Schema Feature

## What is Target Schema Mode?

Instead of letting the system automatically design your output, you can now specify exactly what columns you want in your unified file.

## When to Use It?

**Use Target Mode when:**
- You have specific reporting requirements
- You need output to match an existing template
- You want calculated columns (totals, averages, counts)
- You're feeding data into another system with fixed format

**Use Auto Mode when:**
- You want the system to figure out the best structure
- You're exploring new data
- You don't have format requirements

## How to Use - Quick Steps

### Method 1: Upload Template File

1. Create a CSV or Excel file with just the column headers you want:
   ```
   customer_id, customer_name, total_sales, order_count
   ```

2. Upload your source data files (as usual)

3. Expand "🎯 Advanced: Define Target Schema"

4. Upload your template file in "Option 1"

5. Click "Unify Files Now"

### Method 2: Describe in Text

1. Upload your source data files

2. Expand "🎯 Advanced: Define Target Schema"

3. In "Option 2", type what you need:
   ```
   I need:
   - customer_id
   - customer_name
   - total_sales (sum of all orders)
   - order_count (how many orders)
   - avg_order_value (calculated)
   ```

4. Click "Unify Files Now"

## What Happens Next?

### ✅ Best Case (90%+ of the time)
- System maps your source data to target columns
- Output file has exactly what you requested
- Success message: "✅ Data unified successfully using your target schema!"

### ⚠️ Fallback Case (Challenging requirements)
- System tries 3 times to match your target
- If it can't map properly, automatically switches to auto mode
- You still get unified data (auto-generated schema)
- Warning message: "⚠️ Target schema mapping failed - used auto-generated schema"

### 💡 Why Fallback is Good
- **Your data is never lost**
- **Process always completes**
- **You still get a usable result**
- Learn what's possible from your source data

## Examples

### Example 1: E-commerce Order Data

**You have:**
- customers.csv (customer_id, name, email)
- orders.csv (order_id, customer_id, amount, date)

**You want:**
```
customer_id, customer_name, total_orders, total_spent, first_order, last_order
```

**Result:**
- System groups orders by customer
- Counts orders → total_orders
- Sums amounts → total_spent
- Finds min date → first_order
- Finds max date → last_order

### Example 2: Property Management

**You have:**
- properties.xlsx (property_id, address, size)
- leases.csv (lease_id, property_id, rent, start_date)
- maintenance.csv (work_order_id, property_id, cost)

**You want:**
```
property_id, address, total_rent_collected, maintenance_cost, net_income
```

**Result:**
- System aggregates leases → total_rent_collected
- Sums maintenance → maintenance_cost
- Calculates → net_income (rent - maintenance)

## Tips for Success

### ✅ Do's
- Use clear, descriptive column names
- Specify calculations when needed ("sum of X", "count of Y")
- Match terminology from your source data when possible
- Provide 5-15 target columns (not too few, not too many)

### ❌ Don'ts
- Request columns that can't possibly exist in source
- Use identical names for different concepts
- Provide 50+ columns (system may struggle)
- Use special characters in column names

## Troubleshooting

**Q: Target mode always falls back to auto. Why?**
A: Your target columns might not match your source data. Check:
- Are column names similar to source?
- Can requested calculations be derived?
- Is your description clear?

**Q: Can I see why target mapping failed?**
A: Yes! Check the logs (if you have access) - they show:
- Which columns mapped successfully
- Which columns were unmappable
- Specific feedback for each retry
- Why fallback was triggered

**Q: Will fallback give me the wrong data?**
A: No! Fallback uses the same auto mode that always worked. Your data is correct, just the column names/structure might differ from your target.

**Q: Can I use target mode with one-to-many data?**
A: Yes! The system will:
1. Detect one-to-many relationships
2. Ask for your aggregation preference
3. Apply that strategy while mapping to your target
4. Give you the best of both features

## Advanced Usage

### Combining with One-to-Many

When you have master + detail data:

1. Upload files (e.g., customers + orders)
2. Specify target schema
3. System detects one-to-many
4. Choose aggregation (SUM, AVG, MAX, MIN, COUNT, or Auto-Solve)
5. System applies aggregation + maps to target

### Template File Best Practices

**Good Template:**
```csv
customer_id,customer_name,total_revenue,order_count,avg_order_value
```

**Better Template (with sample data):**
```csv
customer_id,customer_name,total_revenue,order_count,avg_order_value
CUST-001,John Doe,5000.00,10,500.00
```
(System only reads headers, but examples help document intent)

### Text Description Best Practices

**Good:**
```
customer_id
customer_name  
total_revenue
order_count
```

**Better:**
```
I need these columns in my output:
1. customer_id - unique identifier
2. customer_name - full name
3. total_revenue - SUM of all order amounts
4. order_count - COUNT of orders
5. avg_order_value - calculated as total_revenue / order_count
```

## Real-World Workflow

```
┌─────────────────────────────────────┐
│ 1. Upload Source Files              │
│    ✓ customers.csv                  │
│    ✓ orders.csv                     │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│ 2. Define Target (Optional)         │
│    Option A: Upload template.csv    │
│    Option B: Describe in text       │
│    Option C: Skip (auto mode)       │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│ 3. System Processing                │
│    • Validates data                 │
│    • Identifies relationships       │
│    • Maps to target (if provided)   │
│    • Generates & executes code      │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│ 4. Result                           │
│    ✅ master_unified_data.xlsx      │
│                                     │
│    Status:                          │
│    • Target mode: YES/NO            │
│    • Fallback: YES/NO               │
│    • Aggregation: [strategy]        │
└─────────────────────────────────────┘
```

## Getting Help

If target mode isn't working:
1. Try auto mode first (to see what's possible)
2. Review auto-generated output
3. Refine your target based on actual data
4. Try target mode again

**Remember:** Fallback is not a failure - it's a safety net that ensures you always get unified data!

## Summary

| Feature | Auto Mode | Target Mode |
|---------|-----------|-------------|
| **Input Required** | Just upload files | Files + target description |
| **Output Control** | System decides | You decide |
| **Success Rate** | 100% | 90%+ (with fallback) |
| **Use Case** | Exploration | Specific requirements |
| **Learning Curve** | None | Minimal |
| **Flexibility** | Limited | High |

**Bottom Line:** Try target mode when you need specific output format. If it doesn't work, auto mode always has your back!
