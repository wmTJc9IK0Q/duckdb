#include "duckdb/planner/operator/logical_unnest.hpp"

#include "duckdb/common/operator/multiply.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_unnest_expression.hpp"

namespace duckdb {

namespace {

//! Rows assumed per input row when the length of the unnested list cannot be derived at plan time.
//!
//! UNNEST is an expanding operator - it emits one row per list element - so the inherited estimate (the maximum over
//! the children, i.e. a pass-through) is not merely imprecise, it models the wrong direction. Every filter, join and
//! aggregate above the UNNEST then compounds that error, and an operator that expands a scan by two orders of
//! magnitude can end up costed as if it had produced fewer rows than the scan did.
//!
//! The number we would want is the average list length, which is a property of the data:
//!   * The type only bounds it for ARRAY(T, N). LIST(T) is unbounded and says nothing.
//!   * DuckDB collects no statistics about it. BaseStatistics for a LIST carries only the statistics of the child
//!     (element) type - see ListStats - and no part of the storage layer computes an element-count distribution, so
//!     there is nothing to read here even in principle.
//! So outside the exact cases handled in EstimatedExpansionFactor this is a guess, and it is wrong on any given
//! query. The only claim made for it is that it is less wrong than 1.
//!
//! The value is PostgreSQL's: estimate_array_length() returned a hard-coded 10 for every array expression until
//! PostgreSQL 17 taught it to read DECHIST element statistics - the statistic we do not have. It is not tuned to any
//! workload; it is a documented "more than one" borrowed from the only other planner that has had to pick one.
constexpr idx_t DEFAULT_UNNEST_EXPANSION = 10;

//! The exact rows produced per input row by unnesting one expression, where that is derivable at plan time.
optional_idx DerivedExpansionFactor(const Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_UNNEST) {
		return optional_idx();
	}
	// The binder wraps an ARRAY argument in a cast to LIST, so the length we are after can sit below a cast. A cast
	// between two collection types preserves the element count, so we may look through any number of them, checking
	// each level from the outside in.
	const Expression *child = expr.Cast<BoundUnnestExpression>().Child().get();
	for (;;) {
		// A fixed-size array unnests to exactly as many rows as the type is wide.
		auto &child_type = child->GetReturnType();
		if (child_type.id() == LogicalTypeId::ARRAY && !ArrayType::IsAnySize(child_type)) {
			return ArrayType::GetSize(child_type);
		}
		// A constant list - UNNEST([1, 2, 3]), or anything the optimizer folded into one - unnests to its own length.
		if (child->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
			auto &value = child->Cast<BoundConstantExpression>().GetValue();
			if (value.IsNull()) {
				return 0;
			}
			if (value.type().id() == LogicalTypeId::LIST) {
				return ListValue::GetChildren(value).size();
			}
			return optional_idx();
		}
		if (child->GetExpressionClass() != ExpressionClass::BOUND_CAST) {
			return optional_idx();
		}
		auto &source = child->Cast<BoundCastExpression>().Child();
		auto source_id = source.GetReturnType().id();
		if (source_id != LogicalTypeId::LIST && source_id != LogicalTypeId::ARRAY) {
			return optional_idx();
		}
		child = &source;
	}
}

} // namespace

vector<ColumnBinding> LogicalUnnest::GetColumnBindings() {
	auto child_bindings = children[0]->GetColumnBindings();
	for (auto unnest_col_idx : ProjectionIndex::GetIndexes(expressions.size())) {
		child_bindings.emplace_back(unnest_index, unnest_col_idx);
	}
	return child_bindings;
}

void LogicalUnnest::ResolveTypes() {
	types.insert(types.end(), children[0]->types.begin(), children[0]->types.end());
	for (auto &expr : expressions) {
		types.push_back(expr->GetReturnType());
	}
}

vector<TableIndex> LogicalUnnest::GetTableIndex() const {
	return vector<TableIndex> {unnest_index};
}

idx_t LogicalUnnest::EstimatedExpansionFactor() const {
	// Unnesting several expressions at once pads the shorter lists with NULLs up to the longest one, so the operator
	// emits max(length) rows per input row - not their sum, and not their product.
	idx_t factor = 0;
	for (auto &expr : expressions) {
		auto derived = DerivedExpansionFactor(*expr);
		factor = MaxValue(factor, derived.IsValid() ? derived.GetIndex() : DEFAULT_UNNEST_EXPANSION);
	}
	// UNNEST(NULL) and UNNEST([]) really do produce nothing, but a zero cardinality is read as "no statistics" in
	// enough of the planner that it is not worth claiming here; those plans are trivial either way.
	return MaxValue<idx_t>(factor, 1);
}

idx_t LogicalUnnest::ExpandCardinality(idx_t child_cardinality) const {
	idx_t result;
	if (!TryMultiplyOperator::Operation<idx_t, idx_t, idx_t>(child_cardinality, EstimatedExpansionFactor(), result)) {
		return NumericLimits<idx_t>::Maximum();
	}
	return result;
}

idx_t LogicalUnnest::EstimateCardinality(ClientContext &context) {
	if (has_estimated_cardinality) {
		return estimated_cardinality;
	}
	SetEstimatedCardinality(ExpandCardinality(children[0]->EstimateCardinality(context)));
	return estimated_cardinality;
}

string LogicalUnnest::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", unnest_index.index);
	}
#endif
	return LogicalOperator::GetName();
}

} // namespace duckdb
