// Minimal ambient declarations for the subset of the WebAssembly JS API used by
// the Wasm loader. TypeScript only ships these in lib.dom/lib.webworker, which
// this Node-only project intentionally excludes from `lib`.
declare namespace WebAssembly {
  interface Module {
    readonly [Symbol.toStringTag]: "WebAssembly.Module";
  }
  interface Instance {
    readonly exports: Record<string, unknown>;
  }
  class Memory {
    constructor(descriptor: { initial: number; maximum?: number; shared?: boolean });
    readonly buffer: ArrayBuffer;
  }
  const Module: {
    new (bytes: ArrayBuffer | ArrayBufferView): Module;
  };
  const Instance: {
    new (module: Module, importObject?: Record<string, Record<string, unknown>>): Instance;
  };
}
