import { ExtensionContext } from "@lichtblick/suite";

// import { initExamplePanel } from "./ExamplePanel";

// export function activate(extensionContext: ExtensionContext): void {
//   extensionContext.registerPanel({ name: "example-panel", initPanel: initExamplePanel });
// }

import { initUmiLabelPanel } from "./UmiLabelPanel";

export function activate(extensionContext: ExtensionContext): void {
  extensionContext.registerPanel({ name: "umi-label-panel", initPanel: initUmiLabelPanel });
}
