//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_unnest.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

//! LogicalUnnest represents the logical UNNEST operator.
class LogicalUnnest : public LogicalOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_UNNEST;

public:
	explicit LogicalUnnest(TableIndex unnest_index)
	    : LogicalOperator(LogicalOperatorType::LOGICAL_UNNEST), unnest_index(unnest_index) {
	}

	TableIndex unnest_index;

public:
	vector<ColumnBinding> GetColumnBindings() override;
	void Serialize(Serializer &serializer) const override;
	static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);
	vector<TableIndex> GetTableIndex() const override;
	string GetName() const override;
	idx_t EstimateCardinality(ClientContext &context) override;

	//! The number of rows this operator is expected to emit per input row. See the comment on the definition: this is
	//! exact for fixed-size arrays and constant lists, and a heuristic otherwise.
	idx_t EstimatedExpansionFactor() const;
	//! child_cardinality scaled by EstimatedExpansionFactor(), saturating instead of overflowing
	idx_t ExpandCardinality(idx_t child_cardinality) const;

protected:
	void ResolveTypes() override;
};
} // namespace duckdb
