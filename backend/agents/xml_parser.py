"""XML parser for SEC Form D filings."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional

import requests


def parse_sec_filing_xml(filing_url: str, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Fetch and parse SEC Form D XML filing to extract structured data."""
    if not filing_url:
        return {}

    session = session or requests.Session()
    # SEC.gov requires User-Agent header
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = session.get(filing_url, headers=headers, timeout=10)
        response.raise_for_status()
        xml_content = response.text
    except requests.RequestException as e:
        return {}

    return _extract_form_d_data(xml_content)


def _extract_form_d_data(xml_content: str) -> Dict[str, Any]:
    """Extract key fields from Form D XML."""
    extracted: Dict[str, Any] = {}

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        # Fallback to regex if XML is malformed
        return _extract_via_regex(xml_content)

    # Namespace handling (SEC filings use namespaces)
    namespaces = {
        "edgar": "http://www.sec.gov/edgar/document/edgardocument",
        "ed": "http://www.sec.gov/edgar/document/edgardocument",
    }

    # Try to find common Form D fields
    # Total offering amount
    for tag in ["totalOfferingAmount", "totalAmountSold", "totalOffering"]:
        elem = root.find(f".//{tag}")
        if elem is None:
            for ns_prefix, ns_url in namespaces.items():
                elem = root.find(f".//{{{ns_url}}}{tag}")
        if elem is not None and elem.text:
            try:
                extracted["total_offering_amount"] = float(elem.text.replace(",", ""))
            except (ValueError, AttributeError):
                pass

    # Total amount sold
    for tag in ["totalAmountSold", "amountSold"]:
        elem = root.find(f".//{tag}")
        if elem is None:
            for ns_prefix, ns_url in namespaces.items():
                elem = root.find(f".//{{{ns_url}}}{tag}")
        if elem is not None and elem.text:
            try:
                extracted["total_amount_sold"] = float(elem.text.replace(",", ""))
            except (ValueError, AttributeError):
                pass

    # Investors
    investors = []
    for investor_elem in root.findall(".//investor") or root.findall(".//relatedPerson"):
        name = None
        if investor_elem.find("name") is not None:
            name = investor_elem.find("name").text
        elif investor_elem.find("firstName") is not None and investor_elem.find("lastName") is not None:
            name = f"{investor_elem.find('firstName').text} {investor_elem.find('lastName').text}"
        if name:
            investors.append(name.strip())
    if investors:
        extracted["investors"] = investors

    # Minimum investment
    for tag in ["minimumInvestment", "minimumInvestmentAmount"]:
        elem = root.find(f".//{tag}")
        if elem is not None and elem.text:
            try:
                extracted["minimum_investment"] = float(elem.text.replace(",", ""))
            except (ValueError, AttributeError):
                pass

    # If XML parsing didn't yield much, try regex fallback
    if not extracted:
        extracted = _extract_via_regex(xml_content)

    return extracted


def _extract_via_regex(xml_content: str) -> Dict[str, Any]:
    """Fallback regex extraction for Form D data."""
    extracted: Dict[str, Any] = {}

    # Look for dollar amounts
    amount_patterns = [
        r"total\s+offering\s+amount[:\s]*\$?([\d,]+)",
        r"total\s+amount\s+sold[:\s]*\$?([\d,]+)",
        r"offering\s+amount[:\s]*\$?([\d,]+)",
    ]
    for pattern in amount_patterns:
        match = re.search(pattern, xml_content, re.IGNORECASE)
        if match:
            try:
                amount = float(match.group(1).replace(",", ""))
                if "total_offering_amount" not in extracted:
                    extracted["total_offering_amount"] = amount
            except (ValueError, AttributeError):
                pass

    return extracted


def enrich_sec_filing(filing_data: Dict[str, Any], session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Enrich SEC filing metadata with parsed XML data."""
    filing_url = filing_data.get("link") or filing_data.get("filingUrl")
    if not filing_url:
        return filing_data

    # Try to fetch and parse XML
    xml_data = parse_sec_filing_xml(filing_url, session)

    # Merge extracted data
    enriched = {**filing_data}
    if xml_data.get("total_offering_amount"):
        enriched["amount_raised"] = xml_data["total_offering_amount"]
    if xml_data.get("total_amount_sold"):
        enriched["amount_sold"] = xml_data["total_amount_sold"]
    if xml_data.get("investors"):
        enriched["investors"] = xml_data["investors"]
    if xml_data.get("minimum_investment"):
        enriched["minimum_investment"] = xml_data["minimum_investment"]

    return enriched

