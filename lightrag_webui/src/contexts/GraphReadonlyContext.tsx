import { createContext, useContext } from 'react'

/**
 * When true, the graph UI is read-only: property rows render without the
 * edit (pencil) affordance. Used by the standalone /knowledge-graph page.
 */
const GraphReadonlyContext = createContext<boolean>(false)

export const GraphReadonlyProvider = GraphReadonlyContext.Provider

export const useGraphReadonly = () => useContext(GraphReadonlyContext)

export default GraphReadonlyContext
