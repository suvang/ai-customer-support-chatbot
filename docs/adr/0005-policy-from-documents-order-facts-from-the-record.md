# Policy from Support documents, Order facts from the record

Store policy comes from the Support library. Facts about one Order come from that Order record: number, Order status, and dates. A reply may use either source, or apply a retrieved rule to that record, such as Refund eligibility. It may not state an Order status or a policy that was not retrieved. Payments and Customer accounts stay out. This supersedes ADR 0003.
