# Device integration research

This document records what integration mechanisms actually, currently exist
for the consumer glucose-adjacent devices and platforms DiaLog might connect
to, as researched via web search. It is the honesty backstop for
`lib/import/connectors/*`: no connector in this codebase claims to talk to an
API or protocol that isn't listed here as real, and every "not verified"
detail in a connector's code comment is called out again here.

**Date checked: 2026-08-24.** Vendor software and APIs change; re-verify
before relying on anything below for a production integration decision.

---

## Abbott FreeStyle Optium Neo

- **Official API/SDK**: none published for consumers or third-party
  developers.
- **Desktop software export**: the meter pairs with Abbott's desktop
  reporting software (distributed under the Abbott Diabetes Care /
  CoPilot-family tooling) over USB, which can produce a report. We could not
  obtain or locate a verified sample export file or an official
  column-layout specification for this export.
- **USB/BLE protocol**: undocumented publicly; some FreeStyle meters use a
  vendor-specific HID/serial protocol that has been reverse-engineered by
  open-source projects (e.g. glucometerutils), but Optium Neo is not
  confirmed among the well-supported models and Abbott has not published a
  protocol spec.
- **Verdict**: file import only, and even that column layout is unverified.
  `lib/import/connectors/abbott-optium-neo.ts` is honestly implemented as a
  thin wrapper around the generic CSV connector rather than a fabricated
  fixed layout.

Sources: general Abbott partner/developer pages surfaced in search did not
include a Optium Neo file spec; see LibreView section below for the
adjacent, better-documented Abbott export.

## Abbott FreeStyle Libre / LibreView

- **Official API**: Abbott does not publish a public developer API for
  LibreView.
- **File export**: LibreView (libreview.com), the cloud portal Libre
  readers/apps sync to, offers a "Glucose data" CSV download from its
  Reports/download section. The layout is not formally published by Abbott,
  but has been independently documented and reverse-engineered by multiple
  open-source projects. Reported columns include: `ID`, `Device`, `Serial
Number`, `Device Timestamp`, `Record Type`, `Historic Glucose mmol/L` (or
  `mg/dL`), `Scan Glucose mmol/L` (or `mg/dL`), `Non-numeric Rapid-Acting
Insulin`, `Rapid-Acting Insulin (units)`, `Non-numeric Food`,
  `Carbohydrates (grams)`, `Non-numeric Long-Acting Insulin`, `Long-Acting
Insulin (units)`, `Notes`, `Strip Glucose mmol/L` (or `mg/dL)`, `Ketone
mmol/L`, and a few insulin-dose breakdown columns. `Record Type` is
  numeric: `0` = Historic Glucose (periodic sensor reading), `1` = Scan
  Glucose (user-initiated NFC scan), and other values cover insulin/food/note
  rows.
- **File preamble**: the export begins with a short block of device/account
  metadata rows before the real header row, which any parser must skip past.
- **Verdict**: file import, well enough documented by the community to
  implement with reasonable confidence — see
  `lib/import/connectors/abbott-libreview.ts`, which still treats
  unrecognised columns defensively since Abbott itself has not published a
  spec and the layout is known to shift slightly across LibreView versions.

Sources:

- [sourcetable.com — export CSV from FreeStyle Libre](https://sourcetable.com/export-csv/freestyle-libre)
- [github.com/nahog/freestyle-libre-parser-viewer](https://github.com/nahog/freestyle-libre-parser-viewer)
- [github.com/shrugalic/LibreView_to_AppleHealth_converter](https://github.com/shrugalic/LibreView_to_AppleHealth_converter)

## Abbott LibreLinkUp

- **Official public API**: none. LibreLinkUp is the caregiver-sharing
  companion app to LibreLink; Abbott has not published a developer API for
  it.
- **Unofficial API**: LibreLinkUp's mobile app talks to an undocumented
  Abbott "sharing" backend that several open-source projects have
  reverse-engineered (e.g. `FokkeZB/libreview-unofficial`,
  `DRFR0ST/libre-link-unofficial-api`, `DiaKEM/libre-link-up-api-client`).
  These are unofficial, unsupported by Abbott, can break without notice on
  any Abbott-side change, and typically require the user's LibreLinkUp
  account credentials (not OAuth) — a real risk for a production app to take
  on.
- **Official commercial route**: Abbott does offer partner/OAuth-based
  integrations to selected companies (e.g. via aggregators like Thryve, or
  direct Abbott partnerships listed at diabetescare.abbott/partnerships),
  but this requires a business agreement with Abbott, not something an app
  can self-serve.
- **Verdict**: not available as a public, self-serve integration. DiaLog
  does **not** implement a LibreLinkUp connector — building on the
  unofficial reverse-engineered API would mean depending on scraped
  credentials-based access to an undocumented backend, which this codebase
  deliberately avoids. Users on Libre should instead export from LibreView
  (see above).

Sources:

- [github.com/FokkeZB/libreview-unofficial](https://github.com/FokkeZB/libreview-unofficial)
- [github.com/DRFR0ST/libre-link-unofficial-api](https://github.com/DRFR0ST/libre-link-unofficial-api)
- [thryve.health — Abbott FreeStyle Libre integration](https://www.thryve.health/features/connections/abbott-freestyle-libre-integration)
- [diabetescare.abbott/partnerships/integrations](https://www.diabetescare.abbott/partnerships/integrations/en.html)

## Omron devices and Omron Connect

- **Official public API**: no self-serve public developer API found. Some
  digital-health data-aggregation vendors (e.g. MyDataHelps) mention
  "the US Omron API", implying Omron offers _some_ B2B/partner API access,
  but not one that is publicly documented or self-serve.
- **File export**: the Omron Connect mobile app has an in-app export
  feature — from History, use the share icon or the "···" menu on a graph
  screen, choose "Export measurement data", and pick CSV (Excel/PDF are also
  offered in some regions). Column layout is not officially published and
  has been reported to vary by region/app version.
- **Product scope**: Omron's consumer product line is overwhelmingly blood
  pressure monitors and body-composition scales; Omron does not sell a
  blood glucose meter in most markets. Any glucose columns in an Omron-style
  export would be unusual and are handled defensively, not as a verified
  feature.
- **Verdict**: file import only (CSV export from the app), primarily for
  blood pressure and weight — see `lib/import/connectors/omron.ts`.

Sources:

- [omron-healthcare.com/omronconnect-support](https://www.omron-healthcare.com/omronconnect-support)
- [support.mydatahelps.org — Omron blood pressure readings data export](https://support.mydatahelps.org/omron-blood-pressure-readings-data-export)
- [omronwellness.com — OMRON connect FAQ](https://omronwellness.com/mobilefaq/page?id=6997-connectivity%2F29569-how-can-i-transfer-all-the-readings-from-the-device)

## Contour / Ascensia

- **Official public API**: none found for individual/consumer developers.
- **File export / third-party aggregation**: Contour meters (Contour,
  Contour Next, Contour Next EZ/ONE, etc.) sync primarily through **Glooko**,
  a third-party diabetes data platform that has device-compatibility
  agreements with Ascensia and other meter makers. Glooko provides its own
  uploader software and account-based sync/export, not a direct
  Contour-to-file path a user can self-serve without Glooko.
- **Verdict**: not available as a direct, self-serve DiaLog integration.
  A user could export data from Glooko itself (where supported) and DiaLog's
  generic CSV connector can ingest whatever columns that export contains,
  but there is no dedicated Contour/Ascensia connector in this codebase
  because there is no verified, direct file format to target.

Sources:

- [glooko.com/compatibility](https://glooko.com/compatibility/)
- [support.glooko.com — Ascensia CONTOUR compatibility](https://glooko.com/glucose_meter/ascensia-contour-next/)

## OneTouch / LifeScan

- **Official public API**: LifeScan publishes a **OneTouch Reveal**
  developer program (onetouch.com/developer) for healthcare-provider-facing
  integrations, but it requires signing a Non-Disclosure Agreement and a
  connectivity agreement with LifeScan — not a self-serve public API a
  hobby/consumer app can call.
- **File export / third-party aggregation**: like Contour, OneTouch meters
  primarily sync through the OneTouch Reveal app/web account, and third-party
  platforms (Glooko) connect to OneTouch Reveal accounts rather than reading
  a meter file directly. Apple Health is also used as a sync intermediary
  for some newer OneTouch Verio models on iOS.
- **Verdict**: not available as a direct, self-serve DiaLog integration for
  the same reasons as Contour/Ascensia. Data that has made it into Apple
  Health via a OneTouch Verio's Health sync is importable through DiaLog's
  Apple Health connector.

Sources:

- [onetouch.com/developer](https://www.onetouch.com/developer)
- [support.glooko.com — connect OneTouch Reveal account to Glooko](https://support.glooko.com/hc/en-us/articles/9300150541331-How-do-I-connect-my-OneTouch-Reveal-Account-to-Glooko)
- [support.glooko.com — sync OneTouch Verio via Apple Health](https://support.glooko.com/hc/en-us/articles/115003859409-How-do-I-sync-my-LifeScan-OneTouch-Verio-Flex-Verio-Reflect-or-Verio-Sync-meter-with-Glooko-via-Apple-Health-iOS)

## Accu-Chek / Roche

- **Official public API**: none found for individual/consumer developers.
- **App integration**: Roche has partnered its Accu-Chek meters (Guide,
  Guide Me, Aviva Connect, Instant) with **mySugr**, a Roche-owned diabetes
  logging app. Compatible meters pair with mySugr over Bluetooth and upload
  readings automatically into the app; mySugr in turn offers PDF/report
  sharing rather than a documented raw-data file export or API.
- **Roche Diabetes Care Platform**: a separate provider-facing platform for
  clinicians reviewing patient device data (RDCP) — not a consumer/developer
  integration surface.
- **Verdict**: not available as a direct, self-serve DiaLog integration.
  A user's data lives in mySugr, which does not offer a documented consumer
  export format we could verify; DiaLog cannot claim to read it.

Sources:

- [diabetes.roche.com — mySugr connected devices](https://diabetes.roche.com/hcp-us/apps-and-data-management/mysugr-connected-devices)
- [mysugr.com/accuchek](https://www.mysugr.com/accuchek)
- [accu-chek.com — connecting to mySugr](https://www.accu-chek.com/support/faq/getting-started/connecting-to-mysugr)

## Dexcom

- **Official public API**: **yes** — this is the one glucose-device vendor
  in this list with a genuine, documented, self-serve developer API.
  `developer.dexcom.com` publishes the Dexcom API (v3), which is a RESTful,
  OAuth 2.0-based API. The core resource for readings is
  `GET /v3/users/self/egvs` (Estimated Glucose Values), returning readings
  roughly every 5 minutes with trend direction and rate of change, always in
  mg/dL. Requests require `startDate`/`endDate` query parameters with a
  maximum 30-day window. Other resources include events, calibrations,
  alerts, devices, and the user's data range. Data uploaded from the Dexcom
  mobile app is delayed by about 1 hour in the US and about 3 hours outside
  the US/Japan.
- **Requires**: registering a developer app with Dexcom and each user
  completing OAuth consent; sandbox vs. production environments are
  separate.
- **Verdict**: a real future direct-integration candidate — see the summary
  table below. Not implemented as a connector in this PR because this PR's
  scope is file-based import connectors, not live OAuth API integrations;
  building it would need application registration, token storage, and a
  refresh-token flow that belongs outside `lib/import`.

Sources:

- [developer.dexcom.com/docs](https://developer.dexcom.com/docs/)
- [developer.dexcom.com — V3 Endpoint Overview](https://developer.dexcom.com/docs/dexcomv3/endpoint-overview/)
- [developer.dexcom.com — Glossary](https://developer.dexcom.com/docs/dexcom/glossary/)

## Apple Health export

- **Mechanism**: not a live API from DiaLog's perspective — it's a
  user-initiated file export. In the iOS Health app: profile icon > "Export
  All Health Data" produces `export.zip` containing `export.xml` (and
  `export_cda.xml`, a clinical-document variant we don't use).
- **Format**: `export.xml`'s schema is not formally published by Apple, but
  is stable and extensively documented by the community. It is a flat list
  of `<Record type="HKQuantityTypeIdentifier..." sourceName="..."
unit="..." startDate="..." endDate="..." value="..." .../>` elements (plus
  `<Workout>` elements for exercise), where `startDate`/`endDate` use the
  format `YYYY-MM-DD HH:mm:ss ±HHMM`. Relevant identifiers for DiaLog:
  `HKQuantityTypeIdentifierBloodGlucose`,
  `HKQuantityTypeIdentifierDietaryCarbohydrates`,
  `HKQuantityTypeIdentifierBodyMass`, `HKCategoryTypeIdentifierSleepAnalysis`.
  Exports can be very large (hundreds of MB for a multi-year history).
- **Verdict**: file import — see `lib/import/connectors/apple-health.ts`.
  There is no live "Apple Health API" a server-side app like DiaLog can call
  directly; on-device access is via Apple's `HealthKit` framework from a
  native iOS app only, which is out of scope for a server-side importer.

Sources:

- [applehealthdata.com — how to export Apple Health data](https://applehealthdata.com/export-apple-health-data/)
- [aihealthexport.com — Apple Health XML format guide](https://www.aihealthexport.com/guides/apple-health-xml-format)
- [tdda.info — In Defence of XML: exporting and analysing Apple Health data](https://www.tdda.info/in-defence-of-xml-exporting-and-analysing-apple-health-data)

## Google Health Connect

- **Mechanism**: Health Connect is an **on-device Android API/data store**
  (developer.android.com/health-and-fitness/health-connect), the successor
  to Google Fit's fitness-data APIs. It is not a cloud API DiaLog's backend
  can call — it's a permissioned, on-device data broker that native Android
  apps read from and write to, with `BloodGlucoseRecord` as one of its
  documented data types (relation-to-meal, specimen source, value, unit).
  All Health Connect data is stored on-device and encrypted; apps request
  granular per-data-type permission from the user.
- **Verdict**: not available as a server-side file/API integration; a real
  integration would require a native/companion Android app using the Health
  Connect SDK to read `BloodGlucoseRecord`s and hand them to DiaLog's
  backend — out of scope for this file-import subsystem. No connector is
  implemented for it here.

Sources:

- [developer.android.com/health-and-fitness/health-connect](https://developer.android.com/health-and-fitness/health-connect)
- [developer.android.com — BloodGlucoseRecord API reference](https://developer.android.com/reference/android/health/connect/datatypes/BloodGlucoseRecord)
- [android-developers.googleblog.com — Introducing Health Connect](https://android-developers.googleblog.com/2022/05/introducing-health-connect.html)

## Fitbit

- **Official public API**: Fitbit has a Web API, but **glucose and blood
  pressure logging were removed** from it — Fitbit does not expose glucose
  data through its API even for users who logged it in-app historically.
- **Deprecation in progress**: Fitbit has announced the legacy Fitbit Web
  API will be fully decommissioned by **September 2026**, migrating to a
  Google Health API with Google OAuth 2.0 replacing Fitbit's current
  authorization, and mandatory user re-consent (no silent migration).
- **Verdict**: not available. Even setting aside the migration churn,
  Fitbit's API does not carry glucose data at all. No connector implemented.

Sources:

- [community.fitbit.com — Fitbit Web API deprecated?](https://community.fitbit.com/t5/Web-API-Development/Fitbit-Web-API-deprecated/td-p/5657469)
- [thryve.health — Fitbit API deprecation](https://www.thryve.health/blog/fitbit-api-deprecation)
- [community.fitbit.com — Introducing the next phase of the Fitbit Web API](https://community.fitbit.com/t5/Web-API-Development/Introducing-the-next-phase-of-the-Fitbit-Web-API/td-p/5821061)

## Garmin

- **Official public API**: Garmin offers Connect IQ (an on-watch app SDK)
  and Health API/Training API products for licensed partners, but no public
  self-serve glucose data API. Garmin devices are not glucose meters/CGMs
  themselves.
- **CGM data on Garmin devices**: some Garmin Connect IQ watch-face/data-field
  apps display CGM readings by pulling from a third-party CGM cloud service
  (Dexcom Share, or the unofficial LibreLinkUp channel) rather than Garmin
  having any native glucose capability — Garmin is a _display_ surface for
  glucose data sourced elsewhere, not a source of it.
- **Verdict**: not applicable as a glucose data source. Garmin data (steps,
  heart rate, sleep) is out of this importer's scope; no connector
  implemented.

Sources:

- [gluroo.com — blood sugar readings on your smartwatch (Garmin/Gluroo)](https://gluroo.com/blog/glucrew/blood-sugar-readings-smartwatch-gluroo/)

## Nightscout

- **Mechanism**: Nightscout (the open-source "#WeAreNotWaiting" CGM
  remote-monitoring project a user self-hosts or uses a hosted instance of)
  exposes a genuine, documented REST API. The core collection is `entries`,
  reachable at `GET /api/v1/entries.json` (optionally with an API
  token/secret depending on site configuration), returning entries shaped
  as `{ type: "sgv"|"mbg"|..., sgv or mbg: number, date: epoch-ms,
dateString: ISO string, direction, device, ... }`. `sgv` = continuous
  sensor glucose value; `mbg` = meter blood glucose (calibration reading).
  Both are always in mg/dL per the Nightscout API convention.
- **Verdict**: real and well-documented, but it's a live API belonging to
  _the user's own Nightscout site_ (not a vendor DiaLog would register with)
  — for this file-import subsystem we treat a saved `entries.json` response
  as a file import target rather than building live OAuth/token
  credential-management for arbitrary user-hosted Nightscout URLs. See
  `lib/import/connectors/nightscout.ts`.

Sources:

- [github.com/nightscout/documentation — api.rst](https://github.com/nightscout/documentation/blob/master/Nightscout/EN/Technical%20info/api.rst)
- [github.com/ecc1/nightscout — api.go](https://github.com/ecc1/nightscout/blob/main/api.go)

## Browser-to-device communication (WebUSB / Web Bluetooth)

- **Web Bluetooth GATT Glucose Service (0x1808)**: a real, standardized
  Bluetooth GATT service (`glucose_measurement` and
  `glucose_measurement_context` characteristics) exists for glucose meters
  that implement Bluetooth Low Energy's Glucose Profile. Where a meter
  implements it, a browser page **can** in principle read it directly via
  the Web Bluetooth API, no vendor app in between.
- **Browser support is limited**: Web Bluetooth works in Chrome 56+, Edge
  79+, Opera 43+, and Samsung Internet 6.2+ (Android only in practice).
  **Firefox and Safari do not support it at all** — meaning any real-world
  browser-based glucose meter connection excludes a large share of desktop
  and essentially all iOS users (Web Bluetooth is unavailable in iOS Safari
  and any iOS browser, since all iOS browsers are WebKit-based).
- **Most consumer meters are not BLE GATT Glucose Profile devices.** The
  devices covered above overwhelmingly use proprietary sync (vendor app +
  cloud), USB/desktop software, or undocumented BLE profiles, not the
  standard 0x1808 service — Abbott, Roche/Accu-Chek, LifeScan/OneTouch, and
  Ascensia/Contour meters sync through vendor apps rather than exposing
  0x1808 to arbitrary web pages.
- **WebUSB**: technically capable of talking to a USB-attached meter that
  uses a custom (non-HID) USB protocol, but every consumer glucose meter
  observed in this research syncs via a vendor's own desktop/mobile
  software rather than documenting a USB protocol for third parties, so
  WebUSB has nothing standard to target here either.
- **Verdict**: technically possible for the narrow case of a meter that
  implements the standard GATT Glucose Profile and a Chromium-family browser
  on a platform Web Bluetooth supports — but it is not a general solution
  given real device support and Safari/Firefox/iOS exclusion, so DiaLog does
  not build a browser-Bluetooth connector in this PR. File import remains
  the reliable path for every vendor above.

Sources:

- [Web Bluetooth browser support overview](https://www.testmuai.com/learning-hub/web-bluetooth-browser-support/)
- [Bluetooth GATT glucose service definition (sputnikdev mirror)](https://github.com/sputnikdev/bluetooth-gatt-parser/blob/master/src/main/resources/gatt/service/org.bluetooth.service.glucose.xml)
- [Silicon Labs AN982 — BLE glucose sensor application note](https://www.silabs.com/documents/public/application-notes/AN982-Bluetooth-LE-Glucose-Sensor.pdf)

---

## Summary table

| Device / platform                   | Supported today via file import                                             | Possible future direct integration                                                               | Not available                                                                                    |
| ----------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Abbott FreeStyle Optium Neo         | ✅ (unverified layout — thin generic-CSV wrapper)                           | Desktop-software protocol, if Abbott ever documents it                                           | —                                                                                                |
| Abbott FreeStyle Libre / LibreView  | ✅ (community-documented CSV layout)                                        | —                                                                                                | Abbott has no public developer API                                                               |
| Abbott LibreLinkUp                  | —                                                                           | Only via an official Abbott partner/OAuth agreement (e.g. Thryve-style)                          | Public self-serve API — only unofficial, credential-based reverse-engineered clients exist       |
| Omron / Omron Connect               | ✅ (app CSV export; BP + weight, glucose handled defensively)               | Possible if Omron's partner API (implied by MyDataHelps) opens up                                | Public self-serve developer API                                                                  |
| Contour / Ascensia                  | — (no dedicated connector; generic CSV could read a Glooko export)          | Via Glooko's platform, if Glooko offers a documented export/API                                  | Direct meter-to-app path without Glooko                                                          |
| OneTouch / LifeScan                 | — (same as above; Apple Health bridge works via the Apple Health connector) | LifeScan's OneTouch Reveal developer program (NDA + agreement required)                          | Public self-serve developer API                                                                  |
| Accu-Chek / Roche (mySugr)          | —                                                                           | If Roche/mySugr publish a documented export/API                                                  | Public self-serve developer API or documented file export                                        |
| Dexcom                              | — (not built in this PR; file-import scope only)                            | ✅ Real, documented OAuth 2.0 REST API (`/v3/users/self/egvs`)                                   | —                                                                                                |
| Apple Health export                 | ✅ (`export.xml`, community-documented schema)                              | Native iOS HealthKit app for live sync                                                           | Server-side/cloud API — HealthKit is on-device only                                              |
| Google Health Connect               | —                                                                           | Native Android companion app using the Health Connect SDK                                        | Server-side/cloud API — on-device only                                                           |
| Fitbit                              | —                                                                           | —                                                                                                | Glucose/BP explicitly removed from the Fitbit API; API itself being decommissioned by Sept 2026  |
| Garmin                              | —                                                                           | —                                                                                                | No glucose capability; watches only display CGM data sourced elsewhere                           |
| Nightscout                          | ✅ (saved `entries.json` response, as a file)                               | Live per-user OAuth/token integration against the user's own Nightscout site URL                 | —                                                                                                |
| Browser Web Bluetooth (GATT 0x1808) | —                                                                           | Narrow: only for a meter implementing the standard Glucose Profile, in a Chromium-family browser | Not general — excludes Safari/Firefox/iOS entirely, and most consumer meters don't expose 0x1808 |
| Browser WebUSB                      | —                                                                           | Only if a vendor ever documents a USB protocol                                                   | No documented USB protocol found for any covered vendor                                          |

**Bottom line**: file import (CSV/XLSX/XML/JSON from a vendor's own export
feature) is the only integration path that is verifiably real and available
today across essentially every device in this list. Dexcom and Nightscout
are the two exceptions with genuine, documented live APIs — both are strong
future-integration candidates, deliberately not built as live OAuth
integrations in this file-import PR. LibreLinkUp, Fitbit-glucose, and a
"generic Bluetooth glucose meter" web page are the three integrations this
document explicitly says **do not** exist as reliable, general, public
mechanisms — no connector in this codebase pretends otherwise.
